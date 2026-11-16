/**
 * The Plan tab's shot rows with one roving Tab stop (S-O, like the Storyboard list, `useCardTabStops`): only the
 * current row and its fields are in the Tab order; arrows move between rows, Home / End jump to the first / last,
 * Enter goes into the row's title. Focusing a field (a click) makes its row current. 19 shots are about 8 Tab
 * stops, not about 95.
 */
import { useRef, useState, type KeyboardEvent } from "react";
import type { StoryboardProject } from "../../types";
import { PlanShotRow } from "./PlanShotRow";
import { isBareKey } from "./storyboardLogic";
import type { UseStoryboardResult } from "./useStoryboard";

export interface PlanShotListProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  canAsk: boolean;
  onModelMissing: () => void;
}

export function PlanShotList({ story, project, canAsk, onModelMissing }: PlanShotListProps) {
  const list = useRef<HTMLOListElement>(null);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const ids = project.shots.map((shot) => shot.id);
  const tabbableId = currentId && ids.includes(currentId) ? currentId : ids[0];

  function focusRow(id: string | undefined) {
    if (!id) return;
    const row = list.current?.querySelector<HTMLElement>(`[data-plan-shot-id="${id}"]`);
    row?.focus();
    row?.scrollIntoView?.({ block: "nearest" });
  }

  function onKey(event: KeyboardEvent<HTMLOListElement>) {
    const row = event.target as HTMLElement;
    if (!row.matches?.("li.plan-shot") || !isBareKey(event)) return;
    const index = ids.indexOf(row.dataset.planShotId ?? "");
    if (index < 0) return;
    const go = (to: number) => {
      event.preventDefault();
      setCurrentId(ids[to]);
      focusRow(ids[to]);
    };
    switch (event.key) {
      case "ArrowDown":
      case "ArrowRight":
        go(Math.min(ids.length - 1, index + 1));
        break;
      case "ArrowUp":
      case "ArrowLeft":
        go(Math.max(0, index - 1));
        break;
      case "Home":
        go(0);
        break;
      case "End":
        go(ids.length - 1);
        break;
      case "Enter": {
        const field = row.querySelector<HTMLElement>(".plan-shot__title:not(:disabled), .plan-shot__rework:not(:disabled)");
        if (field) {
          event.preventDefault();
          field.focus();
        }
        break;
      }
      default:
        break;
    }
  }

  return (
    <ol
      className="plan-shots__list"
      ref={list}
      onKeyDown={onKey}
      onFocusCapture={(event) => {
        const id = (event.target as HTMLElement).closest<HTMLElement>("[data-plan-shot-id]")?.dataset.planShotId;
        if (id) setCurrentId(id);
      }}
    >
      {project.shots.map((shot, index) => (
        <PlanShotRow
          key={shot.id}
          story={story}
          projectId={project.id}
          shot={shot}
          number={index + 1}
          canAsk={canAsk}
          tabbable={shot.id === tabbableId}
          onModelMissing={onModelMissing}
        />
      ))}
    </ol>
  );
}
