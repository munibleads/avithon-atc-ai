export async function transcribeWithBackend(audioPath: string): Promise<string> {
  try {
    const response = await fetch("http://localhost:8000/transcribe", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ audio_path: audioPath }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch transcription");
    }

    const data = await response.json();
    return data.transcript; // expects { transcript: "..." }
  } catch (error) {
    console.error("Transcription error:", error);
    return "Error during transcription.";
  }
}