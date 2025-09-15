import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { saveAs } from "file-saver";
import * as XLSX from "xlsx";

function CrosstabPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [menuOpen, setMenuOpen] = useState(null);
  const [sortConfigs, setSortConfigs] = useState({});

  // fetch
  const runQuickCrosstab = () => {
    setLoading(true);
    setError(null);
    fetch(`http://127.0.0.1:8000/api/projects/${id}/quick_crosstab/`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to generate crosstab");
        return res.json();
      })
      .then((data) => {
        const arr = Array.isArray(data) ? data : [data];
        const normalized = arr.map((it, idx) => {
          const payload = it.data || it;
          const question =
            it.question ||
            it.Table_Title ||
            it.var_group ||
            it.var_name ||
            `Q${idx + 1}`;
          return { question, data: payload, add_type: it.add_type ?? it.addType ?? "", };
        });
        setGroups(normalized);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Error generating crosstabs");
        setLoading(false);
      });
  };

  const exportSummaryExcel = (summaryObj, keyName = "Summary") => {
    if (!summaryObj) return;
    const attributes = summaryObj.columns || [];
    const rows = summaryObj.rows || [];

    const countAoA = [["Attribute", "Count"]];
    const pctAoA = [["Attribute", "Percent"]];

    attributes.forEach((attrLabel, idx) => {
      const cell = rows[0]?.cells?.[idx] || {};
      countAoA.push([attrLabel, cell.count ?? ""]);
      pctAoA.push([attrLabel, cell.pct ?? ""]);
    });

    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet(countAoA),
      `${keyName}_Count`
    );
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet(pctAoA),
      `${keyName}_Percent`
    );

    const buf = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    saveAs(
      new Blob([buf], { type: "application/octet-stream" }),
      `${keyName}.xlsx`
    );
  };

  // -------- Sorting helpers ----------
  const handleSort = (tableKey, colIndex) => {
    console.log("Sorting", tableKey, colIndex);
    setSortConfigs((prev) => {
      const prevCfg = prev[tableKey] || { colIndex: null, direction: "none" };
      if (prevCfg.colIndex === colIndex) {
        if (prevCfg.direction === "desc")
          return { ...prev, [tableKey]: { colIndex, direction: "asc" } };
        if (prevCfg.direction === "asc")
          return { ...prev, [tableKey]: { colIndex: null, direction: "none" } };
        return { ...prev, [tableKey]: { colIndex, direction: "desc" } };
      } else {
        return { ...prev, [tableKey]: { colIndex, direction: "desc" } };
      }
    });
  };

  const getSortedRows = (tableObj, tableKey) => {
    if (!tableObj?.rows) return [];
    const cfg = sortConfigs[tableKey] || { colIndex: null, direction: "none" };
    const rows = [...tableObj.rows];

    if (!tableObj.columns || tableObj.columns.length <= 1) {
      if (cfg.colIndex !== null && cfg.direction !== "none") {
        return rows.sort((a, b) =>
          cfg.direction === "desc"
            ? (b.count || 0) - (a.count || 0)
            : (a.count || 0) - (b.count || 0)
        );
      }
      return rows;
    }

    const fixedTop = rows.filter((r) => r.label === "Total");
    const fixedBottom = rows.filter((r) =>
      ["Mean", "Median", "StdDev"].includes(r.label)
    );
    let sortable = rows.filter(
      (r) => !["Total", "Mean", "Median", "StdDev"].includes(r.label)
    );

    if (cfg.colIndex !== null && cfg.direction !== "none") {
      sortable = sortable.sort((a, b) => {
        const aVal = a.cells?.[cfg.colIndex]?.count || 0;
        const bVal = b.cells?.[cfg.colIndex]?.count || 0;
        return cfg.direction === "desc" ? bVal - aVal : aVal - bVal;
      });
    }

    return [...fixedTop, ...sortable, ...fixedBottom];
  };


  const renderNPSCard = (npsObj, question, tableKey) => {
    if (!npsObj) return null;
    const cols = ["Qualified"];

    const status =
      npsObj.score >= 50 ? "Excellent"
        : npsObj.score >= 0 ? "Good"
          : "Needs Improvement";

    const u_color =
      npsObj.score >= 50 ? "text-[#FF5578]"
        : npsObj.score >= 0 ? "text-[#F39A63]"
          : "text-[#1BD56F]";

    const markerLeft = ((npsObj.score + 100) / 200) * 100;

    // Split rows into fixed + sortable
    const fixedTop = [{ label: "Total", count: npsObj.base }];
    const fixedBottom = [
      { label: "Mean", count: npsObj.mean },
      { label: "Median", count: npsObj.median },
      { label: "Std Dev", count: npsObj.stddev }
    ];
    let sortable = [
      { label: "NPS Score", count: Math.round(npsObj.score) },
      { label: "Promoter (9–10)", count: npsObj.promoter.count, pct: npsObj.promoter.pct },
      { label: "Passive (7–8)", count: npsObj.passive.count, pct: npsObj.passive.pct },
      { label: "Detractor (0–6)", count: npsObj.detractor.count, pct: npsObj.detractor.pct }
    ];

    // Apply sorting only to sortable rows
    const cfg = sortConfigs[tableKey] || { colIndex: null, direction: "none" };
    if (cfg.colIndex !== null && cfg.direction !== "none") {
      sortable = sortable.sort((a, b) => {
        const aVal = a.count || 0;
        const bVal = b.count || 0;
        return cfg.direction === "desc" ? bVal - aVal : aVal - bVal;
      });
    }

    const sortedRows = [...fixedTop, ...sortable, ...fixedBottom];

    return (
      <div id={tableKey} className="bg-white border rounded-lg shadow p-6 mb-6" key={tableKey}>
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div className="text-green-700 font-semibold">{question}</div>
          <button className="text-sm text-blue-600 hover:underline">Export</button>
        </div>

        {/* Score + status */}
        <div className="flex items-baseline mb-4 flex-col">
          <div className="text-2xl font-bold mr-4">{Math.round(npsObj.score)}</div>
          <div className="text-lg">
            <span className="text-gray-500 italic text-sm">Your Net Promoter Score is </span>
            <u className={`${u_color} underline decoration-2`}>
              <span className="text-black">{status}</span>
            </u>
          </div>
        </div>

        {/* Scale labels */}
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          <span>-100</span>
          <span>0</span>
          <span>100</span>
        </div>

        {/* Gauge */}
        <div className="relative h-4 w-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500 rounded">
          <div
            className="absolute -top-3 w-10 h-10 rounded-full bg-black flex items-center justify-center text-white text-xs font-bold"
            style={{ left: `calc(${markerLeft}% - 20px)` }}
          >
            {Math.round(npsObj.score)}
          </div>
        </div>
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          <span>Needs Improvement</span>
          <span></span>
          <span>Excellent</span>
        </div>

        {/* Table */}
        <table className="min-w-full border-collapse text-xs mt-6">
          <thead className="bg-gray-100">
            <tr>
              <th className="border px-3 py-2 text-left bg-gray-200 min-w-[10vw]">Scale</th>
              {cols.map((c, ci) => {
                const colSort = sortConfigs[tableKey] || {};
                return (
                  <th
                    key={ci}
                    onClick={() => handleSort(tableKey, ci)}
                    className={`border px-3 py-2 text-center cursor-pointer hover:bg-blue-50 min-w-[5vw] ${colSort.colIndex === ci && colSort.direction !== "none"
                      ? "text-blue-600 font-bold underline"
                      : "text-gray-800"
                      }`}
                  >
                    {c}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((r, ri) => (
              <tr
                key={ri}
                className={ri % 2 ? "bg-gray-50 hover:bg-yellow-50" : "bg-white hover:bg-yellow-50"}
              >
                <td className="border px-3 py-2 font-semibold">{r.label}</td>
                <td className="border px-3 py-2 text-center">
                  <div className="flex flex-col items-center">
                    <span>{r.pct || ""}</span>
                    <span className="text-gray-500">{r.count}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };








  // -------- Renderers ----------
  const renderMatrixCard = (matrixObj, groupIdx, tableKeyBase, question) => {
    if (!matrixObj) return null;
    const tableKey = `${tableKeyBase}_Matrix`;
    const cols = matrixObj.columns || ["Qualified"];
    const rows = getSortedRows(matrixObj, tableKey);

    return (
      <div
        id={`matrix_${groupIdx}`}
        className="bg-white border rounded-lg shadow p-4 mb-6"
        key={tableKey}
      >
        <div className="flex justify-between items-center mb-3">
          <div className="text-green-700 font-semibold">{question}</div>
          <button
            onClick={() =>
              exportMatrixExcel(matrixObj, `Matrix_${groupIdx}.xlsx`)
            }
            className="text-sm text-blue-600 hover:underline"
          >
            Export
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-xs">
            <thead className="bg-gray-100">
              <tr>
                <th className="border px-3 py-2 text-left bg-gray-200 min-w-[10vw]">
                  Scale
                </th>
                {cols.map((c, ci) => {
                  const colSort = sortConfigs[tableKey] || {};
                  return (
                    <th
                      key={ci}
                      onClick={() => handleSort(tableKey, ci)}
                      className={`border px-3 py-2 text-center cursor-pointer hover:bg-blue-50 min-w-[5vw] ${colSort.colIndex === ci && colSort.direction !== "none"
                        ? "text-blue-600 font-bold underline"
                        : "text-gray-800"
                        }`}
                    >
                      {c}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr
                  key={ri}
                  className={`${ri % 2 ? "bg-gray-50" : "bg-white"
                    } hover:bg-yellow-50`}
                >
                  <td className="border px-3 py-2 font-semibold">{r.label}</td>
                  {r.cells ? (
                    r.cells.map((cell, ci) => (
                      <td key={ci} className="border px-3 py-2 text-center">
                        <div className="flex flex-col items-center">
                          <span>{cell.pct}</span>
                          <span className="text-gray-500">{cell.count}</span>
                        </div>
                      </td>
                    ))
                  ) : (
                    <td className="border px-3 py-2 text-center">
                      <div className="flex flex-col items-center">
                        <span>{r.pct}</span>
                        <span className="text-gray-500">{r.count}</span>
                      </div>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderSummaryCard = (summaryObj, groupIdx, keyName) => {
    if (!summaryObj) return null;
    const tableKey = `g${groupIdx}_${keyName}`;
    const cols = summaryObj.columns || []; // attributes
    const rows = summaryObj.rows || [];

    // average calculation from first row values
    let avgPct = 0;
    let validCount = 0;
    if (rows[0]?.cells) {
      rows[0].cells.forEach((c) => {
        if (c?.pct && c.pct !== "0%") {
          avgPct += parseFloat(c.pct.replace("%", "")) || 0;
          validCount++;
        }
      });
    }
    const avgPctStr =
      validCount > 0 ? `${(avgPct / validCount).toFixed(1)}%` : "0%";

    return (
      <div
        className="bg-white border rounded-lg shadow p-4 mb-6"
        key={tableKey}
      >
        <div className="flex justify-between items-center mb-3">
          <div className="text-green-700 font-semibold">{keyName}</div>
          <button
            onClick={() => exportSummaryExcel(summaryObj, keyName)}
            className="text-sm text-blue-600 hover:underline"
          >
            Export
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="border px-3 py-2 text-left bg-gray-100">
                  Attribute
                </th>
                <th className="border px-3 py-2 text-center bg-gray-100">
                  Qualified
                </th>
              </tr>
            </thead>
            <tbody>
              {/* --- Average row --- */}
              <tr className="bg-gray-200 font-bold">
                <td className="border px-3 py-2">Average</td>
                <td className="border px-3 py-2 text-center">
                  <span>{avgPctStr}</span>
                </td>
              </tr>

              {/* --- Attribute rows --- */}
              {cols.map((attrLabel, idx) => {
                const cell = rows[0]?.cells?.[idx] || {};
                const linkId = `matrix_${groupIdx}`;   // ✅ just group index, matches matrix card
                return (
                  <tr
                    key={idx}
                    className={`${idx % 2 ? "bg-gray-50" : "bg-white"} hover:bg-yellow-50`}
                  >
                    <td className="border px-3 py-2">
                      <a
                        href={`#${linkId}`}
                        className="text-blue-600 hover:underline"
                      >
                        {attrLabel}
                      </a>
                    </td>
                    <td className="border px-3 py-2 text-center">
                      <div className="flex flex-col items-center">
                        <span>{cell.pct || ""}</span>
                        <span className="text-gray-500">{cell.count || ""}</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // -------- Page render ----------
  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Project Crosstabs</h1>
        <button
          onClick={() => navigate("/")}
          className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Back to Projects
        </button>
      </div>

      <div className="mb-6">
        <button
          onClick={runQuickCrosstab}
          disabled={loading}
          className={`px-4 py-2 rounded-lg ${loading
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700 text-white"
            }`}
        >
          {loading ? "Generating..." : "Run Quick Crosstab"}
        </button>
      </div>

      {loading && (
        <div className="flex justify-center my-6">
          <div className="w-12 h-12 border-4 border-blue-400 border-dashed rounded-full animate-spin"></div>
        </div>
      )}

      {error && <div className="text-red-500 mb-4">{error}</div>}

      <div className="space-y-8">
        {groups.map((grp, gIdx) => {
          const payload = grp.data || {};
          const matrix = payload.Matrix || payload.Total || payload;
          const summaryKeys = [
            "Top Summary",
            "Top2 Summary",
            "Bottom Summary",
            "Bottom2 Summary",
            "Middle Summary",
            "Mean Summary",
          ];


          return (
            <div key={gIdx}>
              {grp.add_type === "NPS"
                ? renderNPSCard(payload.NPS, grp.question, `nps_${gIdx}`)  // 🔹 custom renderer for NPS
                : renderMatrixCard(matrix, gIdx, `g${gIdx}`, grp.question)}

              {summaryKeys.map((k) =>
                payload[k] ? renderSummaryCard(payload[k], gIdx, k) : null
              )}
            </div>
          );


        })}
      </div>
    </div>
  );
}

export default CrosstabPage;