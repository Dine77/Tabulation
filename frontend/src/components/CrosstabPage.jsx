import React, { useState } from "react";
import { useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { saveAs } from "file-saver";
import * as XLSX from "xlsx";
import { Dialog } from "@headlessui/react";


// Modal for New Crosstab
const NewCrosstabModal = ({ open, onClose, topbreaks, selected, setSelected, onRun }) => {
  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="bg-white rounded-lg p-6 w-full max-w-md shadow-xl">
          <Dialog.Title className="text-lg font-bold mb-4">
            Create New Crosstab
          </Dialog.Title>

          <label className="block mb-2 text-sm font-medium text-gray-700">
            Select Topbreak
          </label>
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="w-full border rounded p-2 mb-6"
          >
            <option value="">-- Choose a Topbreak --</option>
            {topbreaks.map((tb, idx) => (
              <option key={idx} value={tb}>{tb}</option>
            ))}
          </select>

          <div className="flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
            >
              Cancel
            </button>
            <button
              onClick={() => onRun(selected)}
              disabled={!selected}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
            >
              Run Crosstab
            </button>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
};


// Main Component
function CrosstabPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [menuOpen, setMenuOpen] = useState(null);
  const [sortConfigs, setSortConfigs] = useState({});

  const [showNewModal, setShowNewModal] = useState(false);
  const [topbreaks, setTopbreaks] = useState([]);
  const [selectedTopbreak, setSelectedTopbreak] = useState("");
  const [sigLevel, setSigLevel] = useState("");

  // Sidebar + Search 
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [touchStartX, setTouchStartX] = useState(0);
  const [touchEndX, setTouchEndX] = useState(0);
  const [activeIndex, setActiveIndex] = useState(null);
  // Scroll to Top Button
  const [showTopButton, setShowTopButton] = useState(false);




  const API_BASE = import.meta.env.VITE_API_URL;


  const baseLabels = ["Base"];
  const summaryLabels = ["Top 2", "Top 3", "Bottom 2", "Bottom 3"];
  const statLabels = ["Mean", "StdDev", "Standard Deviation", "Count"];

  const getRowClass = (label) => {
    if (baseLabels.includes(label)) {
      return "bg-gray-300 font-semibold text-gray-900 hover:bg-yellow-50";
    }
    if (summaryLabels.includes(label)) {
      return "bg-gray-100 text-gray-600 hover:bg-yellow-50";
    }
    if (statLabels.includes(label)) {
      return "bg-gray-200 text-gray-800 hover:bg-yellow-50";
    }
    return "bg-white hover:bg-yellow-50";
  };


  // Scroll Handlers
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 200) {
        setShowTopButton(true);
      } else {
        setShowTopButton(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);
  // Scrollspy for active question
  useEffect(() => {
    const handleScroll = () => {
      let currentIdx = null;
      let minDistance = Infinity;

      groups.forEach((_, idx) => {
        const el = document.getElementById(`matrix_${idx}`);
        if (el) {
          const rect = el.getBoundingClientRect();
          const distance = Math.abs(rect.top - 100); // 100px offset from top
          if (distance < minDistance) {
            minDistance = distance;
            currentIdx = idx;
          }
        }
      });

      if (currentIdx !== null) {
        setActiveIndex(currentIdx);
      }
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll(); // run once on mount

    return () => window.removeEventListener("scroll", handleScroll);
  }, [groups]);




  useEffect(() => {
    if (showNewModal) {
      fetch(`http://127.0.0.1:8000/api/projects/${id}/meta_titles/`)
        .then((res) => res.json())
        .then((data) => {
          setTopbreaks(data); // ✅ fills dropdown
        })
        .catch((err) => console.error("Error fetching topbreaks:", err));
    }
  }, [showNewModal]);




  // fetch topbreak and sig level
  const runNewCrosstab = (topbreak, sigLevel) => {
    setLoading(true);
    setError(null);

    fetch(
      `http://127.0.0.1:8000/api/projects/${id}/new_crosstab/?topbreak=${encodeURIComponent(topbreak)}&sig=${sigLevel}`
    )
      .then((res) => {
        if (!res.ok) throw new Error("Failed to generate new crosstab");
        return res.json();
      })
      .then((data) => {
        const arr = Array.isArray(data) ? data : [data];
        const normalized = arr.map((it, idx) => ({
          ...it,
          question: it.question || it.Table_Title || it.var_name || `Q${idx + 1}`,
          data: it.data || it,
          crosstab_type: "new",   // ✅ mark it so we can distinguish later
        }));
        setGroups(normalized); // replace old tables
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Error generating new crosstabs");
        setLoading(false);
      });
  };



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
          return {
            question,
            data: payload,
            var_group: it.var_group ?? null,   // ✅ preserve group info for SRG/MR
            var_name: it.var_name ?? null,     // ✅ preserve var name
            type: it.type ?? "",               // ✅ keep type (SC, MR, SRG, etc.)
            add_type: it.add_type ?? it.addType ?? "",
          };
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
      ["Mean", "StdDev"].includes(r.label)
    );
    let sortable = rows.filter(
      (r) => !["Total", "Mean", "StdDev"].includes(r.label)
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
    const fixedTop = [{ label: "Base", count: npsObj.base }];
    const fixedBottom = [
      { label: "Mean", count: npsObj.mean },
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
              <th className="border px-3 py-2 text-left bg-gray-200 min-w-[10vw]">
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
                    {c.letter && (
                      <span className="text-blue-600 font-bold mr-1">{c.letter}</span>
                    )}
                    {c.label}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((r, ri) => (
              <tr
                key={ri}
                className={r.label == "Mean" || r.label == "Std Dev" ? "bg-gray-200 hover:bg-yellow-50" : "bg-white hover:bg-yellow-50"}
              >
                <td className="border px-3 py-2 font-semibold">{r.label}</td>
                <td className="border px-3 py-2 text-center">
                  <div className="flex flex-col items-center">
                    <span>
                      {r.pct
                        ? `${Math.round(parseFloat(r.pct))}%`
                        : ""}
                    </span>

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
  // Case 1: backend sends multiple sub-tables (when topbreak is applied)
  const renderMatrixCard = (matrixObj, groupIdx, tableKeyBase, question) => {

    if (!matrixObj) return null;

    // 🔹 Case 1: SRG → array of topbreaks
    if (Array.isArray(matrixObj)) {
      return (
        <div
          id={`matrix_${groupIdx}`}
          className="bg-white border rounded-lg shadow p-4 mb-6"
        >
          <div className="flex justify-between items-center mb-3">
            <div className="text-green-700 font-semibold">{question}</div>
            <button className="text-sm text-blue-600 hover:underline">Export</button>
          </div>

          {matrixObj.map((sub, si) => {
            const matrix = sub.Matrix;
            if (!matrix) return null;

            const tableKey = `${tableKeyBase}_sub${si}`;

            // ✅ matrix.columns are plain strings (Exercise, Reading…)
            const catCols = (matrix.columns || []).map((c) =>
              typeof c === "string" ? { label: c } : c
            );

            const rows = getSortedRows(matrix, tableKey);

            return (
              <div key={si} className="mb-6">
                <div className="text-blue-600 font-bold mb-2">
                  {sub.topbreak_label}
                </div>

                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse text-xs">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="border px-3 py-2 text-left bg-gray-100">Scale</th>
                        {catCols.map((c, ci) => (
                          <th key={ci} className="border px-3 py-2 text-center">
                            {c.letter && (
                              <span className="text-blue-600 font-bold mr-1">{c.letter}</span>
                            )}
                            <span>{c.label?.trim()}</span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r, ri) => (
                        <tr
                          key={ri}
                          className={getRowClass(r.label)}
                        >
                          <td className="border px-3 py-2 font-semibold">{r.label}</td>
                          {r.cells?.map((cell, ci) => (
                            <td key={ci} className="border px-3 py-2 text-center">
                              <div className="flex flex-col items-center">
                                {/* ✅ Percent + sig letters */}
                                <span>
                                  {cell.pct ? `${Math.round(parseFloat(cell.pct))}%` : ""}
                                  {cell.sig && (
                                    <span className="text-red-500 text-[10px] font-bold ml-1">
                                      {cell.sig}
                                    </span>
                                  )}
                                </span>
                                {/* ✅ Count always below */}
                                <span className="text-gray-500">{cell.count}</span>
                              </div>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </div>
      );
    }

    // 🔹 Case 2: Normal → single object
    const tableKey = `${tableKeyBase}_Matrix`;

    // normalize columns
    const cols = (matrixObj.columns || []).map((c) =>
      typeof c === "string" ? { label: c, letter: "" } : c
    );

    const rows = getSortedRows(matrixObj, tableKey);

    return (
      <div
        id={`matrix_${groupIdx}`}
        className="bg-white border rounded-lg shadow p-4 mb-6"
      >
        <div className="flex justify-between items-center mb-3">
          <div className="text-green-700 font-semibold">{question}</div>
          <button className="text-sm text-blue-600 hover:underline">Export</button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-xs">
            <thead className="bg-gray-100">
              <tr>
                <th className="border px-3 py-2 text-left bg-gray-100">Scale</th>
                {cols.map((c, ci) => (
                  <th key={ci} className="border px-3 py-2 text-center">
                    {c.letter && (
                      <span className="text-blue-600 font-bold mr-1">{c.letter}</span>
                    )}
                    <span>{c.label?.trim()}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri} className={getRowClass(r.label)}>
                  <td className="border px-3 py-2 font-semibold">{r.label}</td>
                  {r.cells?.map((cell, ci) => (
                    <td key={ci} className="border px-3 py-2 text-center">
                      <div className="flex flex-col items-center">
                        {/* ✅ Percent + sig letters */}
                        <span>
                          {cell.pct ? `${Math.round(parseFloat(cell.pct))}%` : ""}
                          {cell.sig && (
                            <span className="text-red-500 text-[10px] font-bold ml-1">
                              {cell.sig}
                            </span>
                          )}
                        </span>
                        {/* ✅ Count always below */}
                        <span className="text-gray-500">{cell.count}</span>
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div >
    );
  };

  // -------- Quick Crosstab Renderers ----------
  const renderQuickCrosstabCard = (qcObj, groupIdx, tableKeyBase, question, type) => {
    if (!qcObj) return null;

    // 🔹 Case 1: SRG in Quick Crosstab
    if (type === "SRG" || type === "MRG" && qcObj.Matrix) {
      const matrix = qcObj.Matrix;
      const cols = (matrix.columns || []).map((c) =>
        typeof c === "string" ? { label: c, letter: "" } : c
      );

      return (
        <div
          id={`matrix_${groupIdx}`}
          className="bg-white border rounded-lg shadow p-4 mb-6"
        >
          <div className="flex justify-between items-center mb-3">
            <div className="text-green-700 font-semibold">{question}</div>
            <button className="text-sm text-blue-600 hover:underline">Export</button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse text-xs">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border px-3 py-2 text-left bg-gray-100">Scale</th>
                  {cols.map((c, ci) => (
                    <th key={ci} className="border px-3 py-2 text-center">
                      {c.letter && (
                        <span className="text-blue-600 font-bold mr-1">
                          {c.letter}
                        </span>
                      )}
                      {c.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.rows.map((r, ri) => (
                  <tr key={ri} className={getRowClass(r.label)}>
                    <td className="border px-3 py-2 font-semibold">{r.label}</td>
                    {r.cells.map((cell, ci) => (
                      <td key={ci} className="border px-3 py-2 text-center">
                        <div className="flex flex-col items-center">
                          <span>
                            {cell.pct
                              ? `${Math.round(parseFloat(cell.pct))}%`
                              : ""}
                          </span>

                          <span className="text-gray-500">{cell.count}</span>
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }

    // 🔹 Case 2: SC / MR / NR in Quick Crosstab
    if (type === "SC" || type === "MR" || type === "NR") {
      const rows = qcObj.rows || [];
      return (
        <div
          id={`matrix_${groupIdx}`}
          className="bg-white border rounded-lg shadow p-4 mb-6"
        >
          <div className="flex justify-between items-center mb-3">
            <div className="text-green-700 font-semibold">{question}</div>
            <button className="text-sm text-blue-600 hover:underline">Export</button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse text-xs">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border px-3 py-2 text-left bg-gray-100"></th>
                  <th className="border px-3 py-2 text-center">Qualified</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, ri) => (
                  <tr key={ri} className={getRowClass(r.label)}>
                    <td className="border px-3 py-2 font-semibold">{r.label}</td>
                    <td className="border px-3 py-2 text-center">
                      <div className="flex flex-col items-center">
                        <span>
                          {r.pct
                            ? `${Math.round(parseFloat(r.pct))}%`
                            : ""}
                        </span>

                        <span className="text-gray-500">{r.count}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }

    return null;
  };



  // -------- Summary Renderers ----------
  const renderSummaryCard = (summaryObj, groupIdx, keyName, isTopbreakMode) => {
    if (!summaryObj) return null;
    const tableKey = `g${groupIdx}_${keyName}`;
    const cols = summaryObj.columns || [];
    const rows = summaryObj.rows || [];

    return (
      <div className="bg-white border rounded-lg shadow p-4 mb-6" key={tableKey}>
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
                <th className="border px-3 py-2 text-left bg-gray-100">Attribute</th>
                <th className="border px-3 py-2 text-center bg-gray-100">Qualified</th>
              </tr>
            </thead>
            <tbody>
              {/* ✅ First row → Summary with average */}
              {rows.length > 0 && (
                <tr className="bg-gray-200 font-bold">
                  {/* <td className="border px-3 py-2">{rows[0].label} (Average %)</td> */}
                  {/* <td className="border px-3 py-2 text-center"> */}
                  {/* compute avg across Row1..RowN */}
                  {/* {(() => {
                      let sum = 0, count = 0;
                      rows[0].cells.forEach((c) => {
                        if (c?.pct) {
                          sum += parseFloat(c.pct.replace("%", "")) || 0;
                          count++;
                        }
                      });
                      return count ? Math.round(sum / count) + "%" : "0%";
                    })()} */}
                  {/* </td> */}
                </tr>
              )}

              {/* ✅ Next rows → each attribute Row1/Row2/Row3 */}
              {cols.map((attrLabel, idx) => {
                const cell = rows[0]?.cells?.[idx] || {};
                return (
                  <tr
                    key={idx}
                    className="bg-white hover:bg-yellow-50"

                  >
                    <td className="border px-3 py-2">{attrLabel}</td>
                    <td className="border px-3 py-2 text-center">
                      <div className="flex flex-col items-center">
                        <span>
                          {cell.pct
                            ? `${Math.round(parseFloat(cell.pct))}%`
                            : ""}
                        </span>
                        <span className="text-gray-500">{cell.count ?? ""}</span>
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



  // Merged summary for new crosstab (multiple columns)
  const renderMergedSummaryCard = (summaryObj, groupIdx, keyName) => {
    if (!summaryObj) return null;

    const cols = summaryObj.columns || [];
    const rows = summaryObj.rows || [];

    return (
      <div
        className="bg-white border rounded-lg shadow p-4 mb-6"
        key={`g${groupIdx}_${keyName}`}
      >
        {/* ✅ Header without inline Average */}
        <div className="flex justify-between items-center mb-3">
          <div className="text-green-700 font-semibold">{keyName}</div>
          <button
            onClick={() => exportSummaryExcel(summaryObj, keyName)}
            className="text-sm text-blue-600 hover:underline"
          >
            Export
          </button>
        </div>

        {/* ✅ Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="border px-3 py-2 text-left bg-gray-100">
                  Attribute
                </th>
                {cols.map((c, ci) => (
                  <th
                    key={ci}
                    className="border px-3 py-2 text-center bg-gray-100"
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...rows].sort((a, b) => (a.label === "Average" ? -1 : b.label === "Average" ? 1 : 0))
                .map((r, ri) => (
                  <tr key={ri} className={getRowClass(r.label)}>
                    <td className="border px-3 py-2 font-semibold">{r.label}</td>
                    {r.cells.map((cell, ci) => (
                      <td key={ci} className="border px-3 py-2 text-center">
                        <div className="flex flex-col items-center">
                          <span>
                            {cell.pct
                              ? `${Math.round(parseFloat(cell.pct))}%`
                              : ""}
                          </span>
                          {cell.count !== null && cell.count !== undefined && (
                            <span className="text-gray-500">{cell.count}</span>
                          )}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };


  // Sidebar Component
  const Sidebar = ({ groups, activeIndex }) => {
    const [searchTerm, setSearchTerm] = useState("");   // ⬅️ keep search local

    return (
      <div className="fixed top-0 left-0 h-full w-64 bg-gray-100 border-r shadow-lg z-40 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b">
          <h2 className="font-bold text-gray-700">Questions</h2>
        </div>

        {/* Search */}
        <div className="p-3 border-b">
          <input
            type="text"
            placeholder="Search question..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full border rounded px-2 py-1 text-sm"
          />
        </div>

        {/* Scrollable list */}
        <div className="flex-1 overflow-y-auto p-2">
          {groups
            .filter((g) =>
              g.question.toLowerCase().includes(searchTerm.toLowerCase())
            )
            .map((g, idx) => (
              <div
                key={idx}
                className={`px-3 py-2 cursor-pointer rounded text-sm ${activeIndex === idx ? "bg-blue-200 font-bold" : "hover:bg-blue-100"
                  }`}
                onClick={() => {
                  document
                    .getElementById(`matrix_${idx}`)
                    ?.scrollIntoView({ behavior: "smooth" });
                }}
              >
                {g.question}
              </div>
            ))}
        </div>
      </div>
    );
  };

  // -------- Page render ----------
  return (
    <><Sidebar groups={groups} activeIndex={activeIndex} />


      {showTopButton && (
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          className="fixed bottom-6 right-6 bg-gray-200 text-gray-700 px-3 py-2 rounded shadow-lg hover:bg-gray-300 flex flex-col items-center"
        >
          <span className="text-lg">↑</span>
          <span className="text-xs">Top</span>
        </button>
      )}


      <div className="ml-64 p-8 max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Project Crosstabs</h1>
          <button
            onClick={() => navigate("/")}
            className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Back to Projects
          </button>
        </div>
        <div className="mb-6 flex gap-4">
          <button
            onClick={runQuickCrosstab}
            disabled={loading}
            className={`px-4 py-2 rounded-lg ${loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700 text-white"}`}
          >
            {loading ? "Generating..." : "Run Quick Crosstab"}
          </button>

          <button
            onClick={() => setShowNewModal(!showNewModal)} // toggle visibility
            className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white"
          >
            New Crosstab
          </button>
        </div>

        {/* ✅ Inline dropdown section */}
        {showNewModal && (
          <div className="bg-white border rounded-lg shadow p-4 w-full max-w-md mt-4">
            <label className="block mb-2 text-sm font-medium text-gray-700">
              Select Topbreak
            </label>
            <select
              value={selectedTopbreak}
              onChange={(e) => setSelectedTopbreak(e.target.value)}
              className="w-full border rounded p-2 mb-4"
            >
              <option value="">-- Choose a Topbreak --</option>
              {topbreaks.map((tb, idx) => (
                <option key={idx} value={tb}>{tb}</option>
              ))}
            </select>

            {/* 🔹 New: Sig Test dropdown */}
            <label className="block mb-2 text-sm font-medium text-gray-700">
              Significance Test Level
            </label>
            <select
              value={sigLevel}
              onChange={(e) => setSigLevel(e.target.value)}
              className="w-full border rounded p-2 mb-4"
            >
              <option value="">-- Choose Sig Test Level --</option>
              <option value="80">80%</option>
              <option value="85">85%</option>
              <option value="90">90%</option>
              <option value="95">95%</option>
              <option value="99">99%</option>
              <option value="None">None</option>
            </select>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowNewModal(false)}
                className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowNewModal(false);
                  runNewCrosstab(selectedTopbreak, sigLevel);
                }}
                disabled={!selectedTopbreak || !sigLevel}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                Run Crosstab
              </button>
            </div>
          </div>
        )}




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

            // ✅ detect new crosstab with topbreak
            const isTopbreakMode = Array.isArray(payload.matrix);

            let card;

            if (grp.add_type === "NPS") {
              card = renderNPSCard(payload.NPS, grp.question, `matrix_${gIdx}`);
            }
            else if ((grp.type === "SRG" || grp.type === "MRG") && grp.crosstab_type === "new") {
              // pass the *array* of matrices for SRG
              card = renderMatrixCard(grp.data.matrix, gIdx, `matrix_${gIdx}`, grp.question);
            }
            else if ((grp.type !== "SRG" || grp.type !== "MRG") && grp.crosstab_type === "new") {
              // pass the *object* matrix for normal questions
              card = renderMatrixCard(grp.data, gIdx, `matrix_${gIdx}`, grp.question);
            }
            else if ((grp.type === "SRG" || grp.type === "MRG") && grp.data.Matrix) {
              // Quick Crosstab SRG
              card = renderQuickCrosstabCard(grp.data, gIdx, `matrix_${gIdx}`, grp.question, grp.type);
            }
            else if (grp.type === "SC" || grp.type === "MR" || grp.type === "NR") {
              // Quick Crosstab SC/MR/NR             
              const total = grp.data.Total;
              card = renderQuickCrosstabCard(total, gIdx, `matrix_${gIdx}`, grp.question, grp.type);
            }
            else {
              // Normal (matrix with rows[].cells)
              card = renderMatrixCard(grp.data, gIdx, `matrix_${gIdx}`, grp.question);
            }


            return (
              <div key={gIdx}>
                {card}

                {/* ✅ Quick Crosstab */}
                {!isTopbreakMode &&
                  summaryKeys.map((k) => payload[k] ? renderSummaryCard(payload[k], gIdx, k, isTopbreakMode) : null
                  )}

                {/* ✅ New Crosstab */}
                {isTopbreakMode &&
                  summaryKeys.map((k) => payload[k] ? renderMergedSummaryCard(payload[k], gIdx, k) : null
                  )}
              </div>
            );

          })}
        </div>
      </div></>
  );
}

export default CrosstabPage;