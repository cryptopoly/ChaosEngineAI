import { describe, expect, it } from "vitest";
import { addInputImage, fileNameOf } from "./InputImagePicker";

describe("InputImagePicker helpers", () => {
  it("takes the file name from posix and windows paths", () => {
    expect(fileNameOf("/Users/a/pics/lighthouse.png")).toBe("lighthouse.png");
    expect(fileNameOf("C:\\pics\\fox.jpg")).toBe("fox.jpg");
  });

  it("adds an image once and keeps only the newest when over the limit", () => {
    const a = { path: "/a.png", name: "a.png" };
    const b = { path: "/b.png", name: "b.png" };
    expect(addInputImage([a], a, 4)).toEqual([a]);
    expect(addInputImage([a], b, 4)).toEqual([a, b]);
    expect(addInputImage([a], b, 1)).toEqual([b]);
  });
});
