"use client";

import { useEffect, useRef } from "react";

type FieldCanvasProps = {
  field: number[][];
  title: string;
  maxValue?: number;
};

function colorForValue(value: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, value));

  if (clamped < 0.25) {
    const t = clamped / 0.25;

    return [
      8,
      Math.round(20 + 60 * t),
      Math.round(50 + 100 * t),
    ];
  }

  if (clamped < 0.5) {
    const t = (clamped - 0.25) / 0.25;

    return [
      Math.round(8 + 30 * t),
      Math.round(80 + 100 * t),
      Math.round(150 + 60 * t),
    ];
  }

  if (clamped < 0.75) {
    const t = (clamped - 0.5) / 0.25;

    return [
      Math.round(38 + 190 * t),
      Math.round(180 + 40 * t),
      Math.round(210 - 140 * t),
    ];
  }

  const t = (clamped - 0.75) / 0.25;

  return [
    Math.round(228 + 27 * t),
    Math.round(220 - 70 * t),
    Math.round(70 - 50 * t),
  ];
}

export function FieldCanvas({
  field,
  title,
  maxValue,
}: FieldCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas || field.length === 0) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    const rows = field.length;
    const columns = field[0]?.length ?? 0;

    if (columns === 0) {
      return;
    }

    const absoluteMaximum =
      maxValue ??
      Math.max(
        ...field.flat().map((value) => Math.abs(value)),
      );

    const normalization =
      absoluteMaximum > 0 ? absoluteMaximum : 1;

    const imageData = context.createImageData(
      columns,
      rows,
    );

    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < columns; x += 1) {
        const value = field[y][x] / normalization;

        const [red, green, blue] =
          colorForValue(value);

        const flippedY = rows - 1 - y;
        const index =
          (flippedY * columns + x) * 4;

        imageData.data[index] = red;
        imageData.data[index + 1] = green;
        imageData.data[index + 2] = blue;
        imageData.data[index + 3] = 255;
      }
    }

    context.putImageData(
      imageData,
      0,
      0,
    );
  }, [field, maxValue]);

  return (
    <section className="field-panel">
      <div className="field-heading">
        <h2>{title}</h2>
        <span>
          {field[0]?.length ?? 0} × {field.length}
        </span>
      </div>

      <div className="canvas-shell">
        <canvas
          ref={canvasRef}
          width={field[0]?.length ?? 1}
          height={field.length || 1}
          className="field-canvas"
        />
      </div>
    </section>
  );
}