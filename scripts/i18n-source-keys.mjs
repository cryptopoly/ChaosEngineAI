/**
 * Source-side i18n key collector, shared by ``i18n-validate.mjs``.
 *
 * Parses every ``.ts`` / ``.tsx`` file under ``src/`` with the TypeScript
 * compiler API and returns each *literal* ``t("key", ...)`` call with the
 * namespace it resolves to and the ``defaultValue`` it carries.  A regex
 * cannot do this reliably, because the namespace is a property of the
 * enclosing component (``useTranslation("studio")``), not of the key:
 *
 *   t("studio:voice.x")                       key names its own namespace
 *   t("voice.x", { ns: "studio" })            ``ns`` option
 *   const { t } = useTranslation("studio")    nearest enclosing scope wins
 *   const { t: tCommon } = useTranslation()   aliases resolve the same way
 *   i18n.getFixedT(lng, "chat")               fixed translator
 *   (t: TFunction<"library">) => ...          typed parameter
 *   (t: TFunction) => ...                     namespace unknown (``ns: null``)
 *
 * A bare ``useTranslation()`` falls back to the configured default
 * namespace (``common``).  ``ns: null`` means the translator is a parameter
 * without a namespace in its type, so the call site cannot say where its key
 * lives; callers decide how to treat those (the validator looks the key up
 * in every namespace).
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

export const DEFAULT_NAMESPACE = "common";

const FILE_EXTS = new Set([".ts", ".tsx"]);
const TRANSLATOR_METHODS = new Set(["t"]);
const TRANSLATOR_OBJECTS = new Set(["i18n", "i18next"]);

async function walk(dir, out = []) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".") || entry.name === "node_modules" || entry.name === "locales" || entry.name === "__tests__") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walk(full, out);
    else if (FILE_EXTS.has(path.extname(entry.name)) && !/\.test\.tsx?$/.test(entry.name)) out.push(full);
  }
  return out;
}

/** Load the TypeScript compiler, or null when dependencies are not installed. */
export function loadTypeScript() {
  try {
    return require("typescript");
  } catch {
    return null;
  }
}

/**
 * Collect translator call sites.
 *
 * @returns {Promise<{
 *   calls: Array<{file:string,line:number,ns:string|null,key:string,defaultValue:string|null,optionsDefault:boolean}>,
 *   dynamic: Array<{file:string,line:number,ns:string|null,prefix:string}>,
 *   hints: Array<{file:string,line:number,key:string}>,
 * }>}
 */
export async function collectSourceKeys(srcDir, rootDir = path.dirname(srcDir)) {
  const ts = loadTypeScript();
  if (!ts) throw new Error("typescript is not installed (run npm install)");
  const calls = [];
  const dynamic = [];
  const unhandled = [];
  const hints = [];

  for (const file of await walk(srcDir)) {
    const text = await fs.readFile(file, "utf-8");
    if (!/\bt\(|\.t\(|\bt\s*:\s*TFunction|useTranslation|getFixedT|defaultValue/.test(text)) continue;
    for (const m of text.matchAll(/\/\/\s*i18n-extract:\s*([\w.:-]+)/g)) {
      hints.push({ file: path.relative(rootDir, file).split(path.sep).join("/"), line: text.slice(0, m.index).split("\n").length, key: m[1] });
    }
    const kind = file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
    const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, kind);
    const rel = path.relative(rootDir, file).split(path.sep).join("/");

    const literalOf = (node) => {
      if (!node) return null;
      if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
      if (ts.isParenthesizedExpression(node)) return literalOf(node.expression);
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.PlusToken) {
        const l = literalOf(node.left);
        const r = literalOf(node.right);
        return l !== null && r !== null ? l + r : null;
      }
      return null;
    };

    const nsFromTypeNode = (typeNode) => {
      // TFunction<"library"> / TFunction<["a","b"]>; bare TFunction gives null.
      if (!typeNode || !ts.isTypeReferenceNode(typeNode)) return undefined;
      if (typeNode.typeName.getText(sf) !== "TFunction") return undefined;
      const arg = typeNode.typeArguments?.[0];
      if (!arg) return null;
      if (ts.isLiteralTypeNode(arg) && ts.isStringLiteral(arg.literal)) return arg.literal.text;
      if (ts.isTupleTypeNode(arg) && arg.elements[0]) {
        const first = arg.elements[0];
        if (ts.isLiteralTypeNode(first) && ts.isStringLiteral(first.literal)) return first.literal.text;
      }
      return null;
    };

    const nsFromTranslatorInit = (init) => {
      // useTranslation("ns") / useTranslation(["ns", ...]) / x.getFixedT(lng, "ns")
      if (!init || !ts.isCallExpression(init)) return undefined;
      const callee = init.expression;
      const name = ts.isIdentifier(callee)
        ? callee.text
        : ts.isPropertyAccessExpression(callee)
          ? callee.name.text
          : "";
      if (name === "useTranslation") {
        const a = init.arguments[0];
        if (!a) return DEFAULT_NAMESPACE;
        const lit = literalOf(a);
        if (lit !== null) return lit;
        if (ts.isArrayLiteralExpression(a) && a.elements[0]) return literalOf(a.elements[0]) ?? null;
        return null;
      }
      if (name === "getFixedT") {
        const a = init.arguments[1];
        if (!a) return DEFAULT_NAMESPACE;
        return literalOf(a) ?? null;
      }
      return undefined;
    };

    // Does this binding element / declaration introduce the identifier `name`?
    // Returns undefined (no), or the namespace (string | null).
    const resolveInScope = (scopeNode, name) => {
      const statements = [];
      if (ts.isSourceFile(scopeNode) || ts.isBlock(scopeNode) || ts.isModuleBlock(scopeNode)) {
        statements.push(...scopeNode.statements);
      }
      for (const st of statements) {
        if (!ts.isVariableStatement(st)) continue;
        for (const decl of st.declarationList.declarations) {
          const ns = nsFromTranslatorInit(decl.initializer);
          if (ns === undefined) {
            // const t = i18n.t.bind(...) etc. are not tracked; plain alias of another translator:
            if (ts.isIdentifier(decl.name) && decl.name.text === name && decl.initializer && ts.isIdentifier(decl.initializer)) {
              return { alias: decl.initializer.text };
            }
            continue;
          }
          if (ts.isIdentifier(decl.name) && decl.name.text === name) return ns; // const tt = i18n.getFixedT(...)
          if (ts.isObjectBindingPattern(decl.name)) {
            for (const el of decl.name.elements) {
              const local = el.name.getText(sf);
              if (local === name) return ns;
            }
          }
        }
      }
      if (
        ts.isFunctionLike(scopeNode) &&
        scopeNode.parameters
      ) {
        for (const p of scopeNode.parameters) {
          if (ts.isIdentifier(p.name) && p.name.text === name) {
            const ns = nsFromTypeNode(p.type);
            if (ns !== undefined) return ns;
            // `(t: T = tStudio)`: the default names the translator the callers normally pass.
            if (p.initializer && ts.isIdentifier(p.initializer)) return { alias: p.initializer.text };
            return null;
          }
          if (ts.isObjectBindingPattern(p.name)) {
            for (const el of p.name.elements) {
              if (el.name.getText(sf) !== name) continue;
              // ({ t }: { t: TFunction<"x"> })
              if (p.type && ts.isTypeLiteralNode(p.type)) {
                for (const m of p.type.members) {
                  if (ts.isPropertySignature(m) && m.name.getText(sf) === (el.propertyName ?? el.name).getText(sf)) {
                    const ns = nsFromTypeNode(m.type);
                    return ns === undefined ? null : ns;
                  }
                }
              }
              return null;
            }
          }
        }
      }
      return undefined;
    };

    const resolveNamespace = (callNode, name) => {
      let hops = 0;
      let wanted = name;
      for (let n = callNode.parent; n && hops < 10000; n = n.parent, hops++) {
        const r = resolveInScope(n, wanted);
        if (r === undefined) continue;
        if (typeof r === "object" && r && r.alias) {
          wanted = r.alias;
          continue;
        }
        return r;
      }
      return null;
    };

    const optionNs = (obj) => {
      for (const p of obj.properties) {
        if (ts.isPropertyAssignment(p) && p.name.getText(sf) === "ns") return literalOf(p.initializer);
      }
      return null;
    };

    const keyAlternatives = (node) => {
      // Literal key(s), expanding `cond ? "a" : "b"`.  Template with
      // substitutions yields a dynamic prefix.
      if (ts.isParenthesizedExpression(node)) return keyAlternatives(node.expression);
      if (ts.isConditionalExpression(node)) {
        return [...keyAlternatives(node.whenTrue), ...keyAlternatives(node.whenFalse)];
      }
      const lit = literalOf(node);
      if (lit !== null) return [{ key: lit }];
      if (ts.isTemplateExpression(node)) return [{ prefix: node.head.text }];
      return [];
    };

    const unwrap = (node) => {
      while (ts.isParenthesizedExpression(node) || ts.isAsExpression(node) || ts.isNonNullExpression(node) || ts.isTypeAssertionExpression(node)) {
        node = node.expression;
      }
      return node;
    };

    // Local wrapper: `const t = (key, defaultValue, values?) => i18next.t(`ns:prefix.${key}`, { defaultValue, ...values })`.
    // Returns { prefix, ns, keyIdx, defIdx } or null.
    const wrapperInfo = (fn) => {
      if (!fn || !(ts.isArrowFunction(fn) || ts.isFunctionExpression(fn) || ts.isFunctionDeclaration(fn))) return null;
      const params = fn.parameters.map((p) => (ts.isIdentifier(p.name) ? p.name.text : null));
      let found = null;
      const scan = (n) => {
        if (found) return;
        if (ts.isCallExpression(n) && n.arguments[0] && ts.isTemplateExpression(n.arguments[0])) {
          const tpl = n.arguments[0];
          const callee = unwrap(n.expression);
          const isT =
            (ts.isPropertyAccessExpression(callee) && callee.name.text === "t") ||
            (ts.isIdentifier(callee) && /^t([A-Z]\w*)?$/.test(callee.text));
          if (isT && tpl.templateSpans.length === 1 && tpl.templateSpans[0].literal.text === "" && ts.isIdentifier(tpl.templateSpans[0].expression)) {
            const keyIdx = params.indexOf(tpl.templateSpans[0].expression.text);
            const opts = n.arguments[1];
            if (keyIdx >= 0 && opts && ts.isObjectLiteralExpression(opts)) {
              for (const p of opts.properties) {
                let valueName = null;
                if (ts.isShorthandPropertyAssignment(p) && p.name.text === "defaultValue") valueName = "defaultValue";
                else if (ts.isPropertyAssignment(p) && p.name.getText(sf) === "defaultValue" && ts.isIdentifier(p.initializer)) valueName = p.initializer.text;
                if (valueName && params.indexOf(valueName) >= 0) {
                  found = { head: tpl.head.text, keyIdx, defIdx: params.indexOf(valueName), isMember: ts.isPropertyAccessExpression(callee) };
                }
              }
            }
          }
        }
        ts.forEachChild(n, scan);
      };
      if (fn.body) scan(fn.body);
      return found;
    };

    // Wrapper declarations of this file, found once: name -> [{ scope, info }].
    const wrappers = new Map();
    const indexWrappers = (n) => {
      if (ts.isFunctionDeclaration(n) && n.name) {
        const info = wrapperInfo(n);
        if (info) wrappers.set(n.name.text, [...(wrappers.get(n.name.text) ?? []), { scope: n.parent, info }]);
      } else if (ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.initializer) {
        const info = wrapperInfo(n.initializer);
        if (info) wrappers.set(n.name.text, [...(wrappers.get(n.name.text) ?? []), { scope: n.parent.parent.parent, info }]);
      }
      ts.forEachChild(n, indexWrappers);
    };
    indexWrappers(sf);

    const findWrapper = (callNode, name) => {
      const list = wrappers.get(name);
      if (!list) return null;
      for (let n = callNode.parent; n; n = n.parent) {
        const hit = list.find((w) => w.scope === n);
        if (hit) return hit.info;
      }
      return null;
    };

    const visit = (node) => {
      if (ts.isObjectLiteralExpression(node)) {
        // `{ key: "board.x", defaultValue: "..." }` descriptors, translated elsewhere by the consumer.
        let key = null;
        let dv = null;
        let hasDv = false;
        for (const p of node.properties) {
          if (!ts.isPropertyAssignment(p)) continue;
          const pn = p.name.getText(sf);
          if (pn === "key") key = literalOf(p.initializer);
          if (pn === "defaultValue") {
            hasDv = true;
            dv = literalOf(p.initializer);
          }
        }
        if (key !== null && hasDv && node.parent && !ts.isCallExpression(node.parent)) {
          const line = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
          let ns = null;
          let k = key;
          const colon = key.indexOf(":");
          if (colon > 0) {
            ns = key.slice(0, colon);
            k = key.slice(colon + 1);
          }
          calls.push({ file: rel, line, ns, key: k, defaultValue: dv, optionsDefault: true, descriptor: true });
        }
      }
      if (ts.isCallExpression(node)) {
        const callee = unwrap(node.expression);
        let name = null;
        const wrapper = ts.isIdentifier(callee) ? findWrapper(node, callee.text) : null;
        if (wrapper) {
          const k = literalOf(node.arguments[wrapper.keyIdx]);
          const d = literalOf(node.arguments[wrapper.defIdx]);
          const line = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
          if (k !== null) {
            let ns = null;
            let head = wrapper.head;
            const colon = head.indexOf(":");
            if (colon > 0) {
              ns = head.slice(0, colon);
              head = head.slice(colon + 1);
            }
            calls.push({ file: rel, line, ns, key: head + k, defaultValue: d, optionsDefault: true, wrapper: true });
          }
          ts.forEachChild(node, visit);
          return;
        }
        if (ts.isIdentifier(callee) && /^t([A-Z]\w*)?$|^ts$|^tf$/.test(callee.text)) name = callee.text;
        else if (
          ts.isPropertyAccessExpression(callee) &&
          TRANSLATOR_METHODS.has(callee.name.text) &&
          ts.isIdentifier(callee.expression)
        ) {
          name = TRANSLATOR_OBJECTS.has(callee.expression.text) ? callee.expression.text + ".t" : callee.expression.text + ".t";
        }
        const first = node.arguments[0];
        if (!name && first) {
          const objArg = node.arguments.find((a) => ts.isObjectLiteralExpression(a) && a.properties.some((p) => p.name && p.name.getText(sf) === "defaultValue"));
          if (objArg) {
            unhandled.push({ file: rel, line: sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1, callee: callee.getText(sf).slice(0, 60), first: first.getText(sf).slice(0, 80) });
          }
        }
        if (name && first) {
          const line = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
          const second = node.arguments[1];
          let defaultValue = null;
          let optionsDefault = false;
          let explicitNs = null;
          if (second) {
            const lit = literalOf(second);
            if (lit !== null) {
              defaultValue = lit;
            } else if (ts.isObjectLiteralExpression(second)) {
              explicitNs = optionNs(second);
              for (const p of second.properties) {
                if (ts.isPropertyAssignment(p) && p.name.getText(sf) === "defaultValue") {
                  optionsDefault = true;
                  defaultValue = literalOf(p.initializer);
                } else if (ts.isShorthandPropertyAssignment(p) && p.name.text === "defaultValue") {
                  optionsDefault = true;
                }
              }
            }
          }
          const alts = keyAlternatives(first);
          if (alts.length) {
            const isMember = name.includes(".");
            let baseNs;
            if (explicitNs) baseNs = explicitNs;
            else if (name === "i18n.t" || name === "i18next.t") baseNs = DEFAULT_NAMESPACE;
            else if (isMember) baseNs = null;
            else baseNs = resolveNamespace(node, name);
            for (const alt of alts) {
              const raw = alt.key ?? alt.prefix;
              let ns = baseNs;
              let key = raw;
              const colon = raw.indexOf(":");
              if (colon > 0) {
                ns = raw.slice(0, colon);
                key = raw.slice(colon + 1);
              }
              if (alt.key !== undefined) calls.push({ file: rel, line, ns, key, defaultValue, optionsDefault });
              else dynamic.push({ file: rel, line, ns, prefix: key });
            }
          }
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(sf);
  }
  return { calls, dynamic, unhandled, hints };
}

/**
 * Give calls whose translator carries no namespace (`ns: null`: a `t`
 * passed as a plain function parameter, or a `{ key, defaultValue }`
 * descriptor translated elsewhere) a namespace, so they can be checked
 * like the rest:
 *
 *   1. a namespace whose English catalog already has the key;
 *   2. otherwise the namespace that owns the key's top-level segment
 *      (`board.*` is Studio's); when several do (`voice.*` is in Studio and
 *      Chat), the one most of the file's resolved calls use, else the
 *      folder's, else the parent folder's.
 *
 * `catalogs` maps namespace to a Map of leaf key to English text.  A call
 * that still has no namespace keeps `ns: null` and is the caller's to report.
 */
export function inferNamespaces(calls, catalogs) {
  const owners = new Map();
  for (const [ns, leaves] of Object.entries(catalogs)) {
    for (const leaf of leaves.keys()) {
      const top = leaf.split(".")[0];
      if (!owners.has(top)) owners.set(top, new Set());
      owners.get(top).add(ns);
    }
  }
  const tally = (match, allowed) => {
    const counts = new Map();
    for (const c of calls) if (c.ns && (!allowed || allowed.has(c.ns)) && match(c)) counts.set(c.ns, (counts.get(c.ns) ?? 0) + 1);
    let best = null;
    for (const [ns, n] of counts) if (!best || n > best[1]) best = [ns, n];
    return best ? best[0] : null;
  };
  for (const c of calls.filter((o) => !o.ns)) {
    const have = Object.keys(catalogs).filter((ns) => catalogs[ns].has(c.key));
    if (have.length) {
      c.ns = have[0];
      continue;
    }
    const candidates = owners.get(c.key.split(".")[0]);
    if (candidates && candidates.size === 1) {
      c.ns = [...candidates][0];
      c.inferred = true;
      continue;
    }
    let ns = tally((o) => o.file === c.file, candidates);
    for (let dir = path.posix.dirname(c.file); !ns && dir && dir !== "." && dir !== "src"; dir = path.posix.dirname(dir)) {
      ns = tally((o) => o.file.startsWith(dir + "/"), candidates);
    }
    if (ns) {
      c.ns = ns;
      c.inferred = true;
    }
  }
  return calls;
}
