/**
 * Pure rules for the Projects list and the project shell (D-65): tab keys, list search and sort, and the
 * remembered sort. No React here so the rules are unit-tested directly.
 */

/** The tab to move to after a key in a tab row, or null when the key is not handled. Wraps around. */
export function nextTabIndex(current: number, count: number, key: string): number | null {
  if (count <= 0) return null;
  switch (key) {
    case "ArrowRight":
      return (current + 1) % count;
    case "ArrowLeft":
      return (current - 1 + count) % count;
    case "Home":
      return 0;
    case "End":
      return count - 1;
    default:
      return null;
  }
}

export const LIST_SORTS = ["edited", "name", "created", "length"] as const;
export type ListSort = (typeof LIST_SORTS)[number];

/** What a card offers the search and the sort. */
export interface SortableCard {
  name: string;
  recency: number;
  createdAt: number;
  seconds: number;
}

/** Cards whose name holds every word of the query (case and accents ignored), in the chosen order. */
export function filterAndSort<T extends SortableCard>(cards: readonly T[], query: string, sort: ListSort, locale?: string): T[] {
  const fold = (text: string) => text.normalize("NFD").replace(/\p{M}/gu, "").toLocaleLowerCase(locale);
  const words = fold(query).split(/\s+/).filter(Boolean);
  const matched = cards.filter((card) => {
    const name = fold(card.name);
    return words.every((word) => name.includes(word));
  });
  const byName = (a: T, b: T) => a.name.localeCompare(b.name, locale, { sensitivity: "base", numeric: true });
  const sorted = matched.slice();
  switch (sort) {
    case "name":
      sorted.sort(byName);
      break;
    case "created":
      sorted.sort((a, b) => b.createdAt - a.createdAt || byName(a, b));
      break;
    case "length":
      sorted.sort((a, b) => b.seconds - a.seconds || byName(a, b));
      break;
    default:
      sorted.sort((a, b) => b.recency - a.recency || byName(a, b));
  }
  return sorted;
}

const SORT_KEY = "chaos.storyboard.listSort";

export function readListSort(): ListSort {
  try {
    const value = window.localStorage.getItem(SORT_KEY);
    return (LIST_SORTS as readonly string[]).includes(value ?? "") ? (value as ListSort) : "edited";
  } catch {
    return "edited";
  }
}

export function writeListSort(sort: ListSort): void {
  try {
    window.localStorage.setItem(SORT_KEY, sort);
  } catch {
    // not remembered
  }
}

/** True for a dragged file that looks like a project bundle. */
export function isBundleName(name: string): boolean {
  return /\.(chaosfilm|zip)$/i.test(name);
}
