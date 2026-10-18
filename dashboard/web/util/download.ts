export async function downloadCheckpoint(id: string, staged: boolean) {
  if (!staged) {
    const response = await fetch(`/api/checkpoints/${id}/stage`, { method: "POST" });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
    }
  }
  window.location.assign(`/api/checkpoints/${id}/download`);
}
