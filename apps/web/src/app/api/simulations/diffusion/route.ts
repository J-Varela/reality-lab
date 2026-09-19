import { NextResponse } from "next/server";

const realityApiUrl =
  process.env.REALITY_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  try {
    const payload = await request.json();

    const response = await fetch(
      `${realityApiUrl}/simulations/diffusion`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        cache: "no-store",
      },
    );

    const data = await response.json();

    return NextResponse.json(data, {
      status: response.status,
    });
  } catch {
    return NextResponse.json(
      {
        error:
          "Reality Lab API is unavailable. Make sure FastAPI is running.",
      },
      {
        status: 502,
      },
    );
  }
}