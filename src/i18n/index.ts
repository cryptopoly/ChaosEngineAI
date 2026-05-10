// eslint-disable-next-line @typescript-eslint/require-await
export async function t(key: string, fallback?: string): Promise<string> {
  const map = await import("./zh-CN");
  return map.default[key] ?? fallback ?? key;
}
