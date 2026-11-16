import type { APIRequestContext } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, TOKEN, openProject } from "./helpers";

// S-O item 1: typed text whose save is refused is never silently replaced. Two windows edit one project (the save
// of the second meets a newer revision), and the backend goes away while a field is being edited.

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

interface Project {
  id: string;
  revision: number;
  plan?: { brief?: string } | null;
  shots: Array<{ id: string; stillPrompt: string }>;
}

async function create(request: APIRequestContext, name: string): Promise<Project> {
  const response = await request.post(`${API_BASE}/api/storyboard/projects`, { headers, data: { name } });
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { project: Project }).project;
}

async function load(request: APIRequestContext, id: string): Promise<Project> {
  const response = await request.get(`${API_BASE}/api/storyboard/projects/${id}`, { headers });
  return ((await response.json()) as { project: Project }).project;
}

test("Two windows: the refused brief stays with Put it back, and Put it back saves it", async ({ page, request }) => {
  const project = await create(request, "Two windows");
  const other = await page.context().newPage();
  await openProject(page, "Two windows", "Plan");
  await openProject(other, "Two windows", "Plan");

  // Window B starts typing first; window A saves its brief meanwhile, so B's copy of the project is stale.
  const briefB = other.getByLabel("Brief");
  await briefB.click();
  await briefB.fill("B: a quiet harbour at dawn.");
  const briefA = page.getByLabel("Brief");
  await briefA.fill("A: a busy market at noon.");
  await briefA.blur();
  await expect.poll(async () => (await load(request, project.id)).plan?.brief).toBe("A: a busy market at noon.");

  // B leaves the field: its save meets the newer revision. B shows the latest, and its own text with Put it back.
  await briefB.blur();
  await expect(other.getByText("Changed elsewhere; showing the latest")).toBeVisible();
  await expect(briefB).toHaveValue("A: a busy market at noon.");
  const notice = other.locator(".unsaved-text");
  await expect(notice).toContainText("Your text was not saved");
  await expect(notice).toContainText("B: a quiet harbour at dawn.");

  await notice.getByRole("button", { name: "Put it back" }).click();
  await expect(briefB).toHaveValue("B: a quiet harbour at dawn.");
  await expect(notice).toHaveCount(0);
  await expect.poll(async () => (await load(request, project.id)).plan?.brief).toBe("B: a quiet harbour at dawn.");
  await other.close();
});

test("Backend gone mid-edit: the typed prompt comes back with Retry once the backend answers again", async ({ page, request }) => {
  const created = await create(request, "Backend gone");
  const applied = await request.post(`${API_BASE}/api/storyboard/projects/${created.id}/ops`, {
    headers,
    data: { revision: created.revision, op: "apply_plan", args: { mode: "replace", shots: [{ title: "Harbour", stillPrompt: "a harbour", motionPrompt: "", seconds: 4 }] } },
  });
  expect(applied.ok()).toBeTruthy();
  await openProject(page, "Backend gone", "Plan");
  const prompt = page.getByLabel("Picture prompt of shot 1");
  await prompt.fill("a harbour at dawn, gulls over the boats");

  // The backend stops answering (every request to it fails) before the field is left.
  const api = `${API_BASE}/**`;
  await page.route(api, (route) => route.abort("connectionrefused"));
  await prompt.blur();
  // Either the field shows the refusal at once, or the tab waits for the backend; the text must survive both.
  await expect(page.locator(".unsaved-text").or(page.getByText("Waiting for the backend…"))).toBeVisible({ timeout: 20_000 });

  await page.unroute(api);
  const notice = page.locator(".unsaved-text");
  await expect(notice).toContainText("Your text was not saved", { timeout: 20_000 });
  await expect(notice).toContainText("a harbour at dawn, gulls over the boats");
  expect((await load(request, created.id)).shots[0].stillPrompt).toBe("a harbour");

  await notice.getByRole("button", { name: "Retry" }).click();
  await expect(notice).toHaveCount(0);
  await expect(page.getByLabel("Picture prompt of shot 1")).toHaveValue("a harbour at dawn, gulls over the boats");
  await expect.poll(async () => (await load(request, created.id)).shots[0].stillPrompt).toBe("a harbour at dawn, gulls over the boats");
});
