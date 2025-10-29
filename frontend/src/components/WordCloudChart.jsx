import React, { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import "echarts-wordcloud";

const WordCloudChart = ({ data, question, chartId }) => {
    const chartRef = useRef(null);
    const [shape, setShape] = useState(localStorage.getItem(`shape_${chartId}`) || "circle");
    const [viewMode, setViewMode] = useState("cloud");
    const [chartKey, setChartKey] = useState(0); // 🔹 key-based reinit

    const shapes = ["circle", "cardioid", "diamond", "star", "triangle", "pentagon"];

    useEffect(() => {
        if (!data || data.length === 0 || !chartRef.current || viewMode !== "cloud") return;

        // ✅ Always dispose previous instance safely before re-render
        echarts.dispose(chartRef.current);

        const chart = echarts.init(chartRef.current);
        const option = {
            title: {
                text: question || "Word Cloud",
                left: "center",
                textStyle: { fontSize: 18, color: "#2563eb", fontWeight: "bold" },
            },
            tooltip: { show: true, formatter: (item) => `${item.name}: ${item.value}` },
            series: [
                {
                    type: "wordCloud",
                    shape,
                    gridSize: 2,
                    sizeRange: [12, 60],
                    rotationRange: [-45, 90],
                    textStyle: {
                        fontFamily: "Poppins, sans-serif",
                        fontWeight: "bold",
                        color: () => {
                            const r = Math.round(Math.random() * 160);
                            const g = Math.round(Math.random() * 160);
                            const b = Math.round(Math.random() * 160);
                            return `rgb(${r}, ${g}, ${b})`;
                        },
                    },
                    emphasis: { textStyle: { shadowBlur: 10, shadowColor: "#333" } },
                    data,
                },
            ],
        };

        chart.setOption(option);
        const resizeHandler = () => chart.resize();
        window.addEventListener("resize", resizeHandler);

        return () => {
            window.removeEventListener("resize", resizeHandler);
            if (!chart.isDisposed()) chart.dispose();
        };
    }, [data, shape, question, viewMode, chartKey]);

    const handleShapeChange = (e) => {
        const newShape = e.target.value;
        setShape(newShape);
        localStorage.setItem(`shape_${chartId}`, newShape);
        setChartKey((k) => k + 1); // 🔹 trigger reinit safely
    };

    const handleExportChart = () => {
        if (!chartRef.current) return;
        const chart = echarts.getInstanceByDom(chartRef.current);
        if (!chart) return;
        const url = chart.getDataURL({ type: "png", backgroundColor: "#fff", pixelRatio: 2 });
        const link = document.createElement("a");
        link.href = url;
        link.download = `${question.replace(/\s+/g, "_")}_wordcloud.png`;
        link.click();
    };

    const handleExportCSV = () => {
        const csv = ["Word,Count"];
        data.forEach((d) => csv.push(`${d.name},${d.value}`));
        const blob = new Blob([csv.join("\n")], { type: "text/csv" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${question.replace(/\s+/g, "_")}_data.csv`;
        link.click();
    };

    const toggleView = () => {
        // ✅ Safe reinit key bump when toggling modes
        setViewMode((prev) => (prev === "cloud" ? "table" : "cloud"));
        setChartKey((k) => k + 1);
    };

    return (
        <div
            id={`wordcloud_container_${chartId}`}
            style={{
                position: "relative",
                width: "100%",
                background: "#fff",
                borderRadius: "12px",
                boxShadow: "0 0 8px rgba(0,0,0,0.1)",
                padding: "20px",
                margin: "20px 0",
            }}
        >
            {/* Toolbar */}
            <div
                style={{
                    position: "absolute",
                    top: "10px",
                    right: "15px",
                    display: "flex",
                    gap: "10px",
                    zIndex: 2,
                }}
            >
                {viewMode === "cloud" && (
                    <select
                        value={shape}
                        onChange={handleShapeChange}
                        style={{
                            padding: "6px 10px",
                            borderRadius: "6px",
                            border: "1px solid #ddd",
                            fontSize: "13px",
                            backgroundColor: "#f9fafb",
                        }}
                    >
                        {shapes.map((s) => (
                            <option key={s} value={s}>
                                {s.charAt(0).toUpperCase() + s.slice(1)}
                            </option>
                        ))}
                    </select>
                )}

                <button
                    onClick={toggleView}
                    style={{
                        padding: "6px 10px",
                        borderRadius: "6px",
                        background: "#e2e8f0",
                        border: "none",
                        fontSize: "13px",
                        cursor: "pointer",
                    }}
                >
                    {viewMode === "cloud" ? "View Data" : "View Cloud"}
                </button>

                {viewMode === "cloud" ? (
                    <button
                        onClick={handleExportChart}
                        style={{
                            padding: "6px 10px",
                            borderRadius: "6px",
                            background: "#2563eb",
                            color: "white",
                            border: "none",
                            fontSize: "13px",
                            cursor: "pointer",
                        }}
                    >
                        Export PNG
                    </button>
                ) : (
                    <button
                        onClick={handleExportCSV}
                        style={{
                            padding: "6px 10px",
                            borderRadius: "6px",
                            background: "#16a34a",
                            color: "white",
                            border: "none",
                            fontSize: "13px",
                            cursor: "pointer",
                        }}
                    >
                        Export CSV
                    </button>
                )}
            </div>

            {/* View Mode Switch */}
            {viewMode === "cloud" ? (
                <div
                    key={chartKey} // 🔹 new key forces clean DOM node
                    ref={chartRef}
                    style={{ width: "100%", height: "500px" }}
                ></div>
            ) : (
                <div
                    key={chartKey} // 🔹 clean node for data view too
                    style={{
                        height: "500px",
                        overflowY: "auto",
                        marginTop: "40px",
                    }}
                >
                    <table
                        style={{
                            width: "100%",
                            borderCollapse: "collapse",
                            textAlign: "left",
                            fontSize: "13px",
                        }}
                    >
                        <thead>
                            <tr style={{ background: "#f1f5f9" }}>
                                <th style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>
                                    Word
                                </th>
                                <th style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>
                                    Count
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((d, i) => (
                                <tr key={i}>
                                    <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>
                                        {d.name}
                                    </td>
                                    <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>
                                        {d.value}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default WordCloudChart;
