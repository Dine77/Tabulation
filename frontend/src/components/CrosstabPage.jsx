import React, { useState } from "react";
import { useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { saveAs } from "file-saver";
import * as XLSX from "xlsx-js-style";
import { Dialog } from "@headlessui/react";
import * as echarts from "echarts";
import "echarts-wordcloud";
import WordCloudChart from "./WordCloudChart";


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
      fetch(`https://apicrosstab.nnet-dataviz.com/api/projects/${id}/meta_titles/`)
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
      `https://apicrosstab.nnet-dataviz.com/api/projects/${id}/new_crosstab/?topbreak=${encodeURIComponent(topbreak)}&sig=${sigLevel}`
    )
      .then((res) => {
        if (!res.ok) throw new Error("Failed to generate new crosstab");
        return res.json();
      })
      .then((data) => {
        const arr = Array.isArray(data) ? data : [data];
        const normalized = arr.map((it, idx) => ({
          ...it,
          question: it.question || it.Table_Title || it.var_name || it.sortorder || `Q${idx + 1}`,
          data: it.data || it,
          crosstab_type: "new",
          sortorder: it.sortorder   // ✅ mark it so we can distinguish later
        }));
        const newdata = normalized.sort((a, b) => a.sortorder - b.sortorder);
        setGroups(newdata); // replace old tables
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
    fetch(`https://apicrosstab.nnet-dataviz.com/api/projects/${id}/quick_crosstab/`)
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
            it.sortorder ||
            `Q${idx + 1}`;
          return {
            question,
            data: payload,
            var_group: it.var_group ?? null,   // ✅ preserve group info for SRG/MR
            var_name: it.var_name ?? null,     // ✅ preserve var name
            type: it.type ?? "",               // ✅ keep type (SC, MR, SRG, etc.)
            add_type: it.add_type ?? it.addType ?? "",
            sortorder: it.sortorder
          };
        });
        const newdata = normalized.sort((a, b) => a.sortorder - b.sortorder);
        setGroups(newdata);
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



  function exportToExcel(divId, data = [], question = "Report") {
    try {
      if (!Array.isArray(data) || data.length === 0) {
        console.warn("⚠️ No data to export");
        return;
      }

      const exportData = [];
      const percentCells = {}; // track which cells are % values
      const countCells = {}; // track count column to force numeric

      // 🟩 Title row (merged later)
      exportData.push([question]);
      exportData.push([]); // blank row

      // 🟩 Header row
      exportData.push(["Label", "Percentage", "Count"]);

      // 🟩 Data rows
      data.forEach((row, idx) => {
        const rowIndex = idx + 3; // offset (title + blank + header = 3)
        const pctVal = row.pct ? parseFloat(row.pct.replace("%", "")) / 100 : "";
        const countVal =
          typeof row.count === "number" && !isNaN(row.count)
            ? row.count
            : row.count ?? "";

        const rowData = [row.label || "", pctVal, countVal];
        exportData.push(rowData);

        // mark percentage cell
        if (pctVal !== "") {
          const cellRef = XLSX.utils.encode_cell({ r: rowIndex, c: 1 });
          percentCells[cellRef] = true;
        }

        // mark count cell (always numeric)
        if (countVal !== "") {
          const cellRef = XLSX.utils.encode_cell({ r: rowIndex, c: 2 });
          countCells[cellRef] = true;
        }
      });

      // Create worksheet
      const ws = XLSX.utils.aoa_to_sheet(exportData);

      // 🟢 Merge the question title row across 3 columns
      ws["!merges"] = [{ s: { r: 0, c: 0 }, e: { r: 0, c: 2 } }];

      // 🟢 Define styles
      const titleStyle = {
        font: { bold: true, sz: 13, color: { rgb: "1E4D2B" } },
        alignment: { horizontal: "left", vertical: "center" },
      };
      const headerStyle = {
        font: { bold: true },
        fill: { fgColor: { rgb: "D9E1F2" } },
        alignment: { horizontal: "center" },
      };
      const baseStyle = {
        font: { bold: true },
        fill: { fgColor: { rgb: "E7E6E6" } },
      };
      const totalStyle = { font: { bold: true } };

      // 🟢 Apply styles + number formatting
      const range = XLSX.utils.decode_range(ws["!ref"]);
      for (let R = range.s.r; R <= range.e.r; ++R) {
        for (let C = range.s.c; C <= range.e.c; ++C) {
          const cellRef = XLSX.utils.encode_cell({ r: R, c: C });
          if (!ws[cellRef]) continue;

          const cell = ws[cellRef];
          cell.s = cell.s || {};

          // Title
          if (R === 0) cell.s = titleStyle;
          // Header
          if (R === 2) cell.s = headerStyle;
          // Base
          if (data[R - 3]?.label?.toLowerCase().includes("base")) {
            cell.s = baseStyle;
          }
          // Total
          if (data[R - 3]?.label?.toLowerCase().includes("total")) {
            cell.s = totalStyle;
          }

          // ✅ Apply % format only to Percentage column (B)
          if (percentCells[cellRef]) {
            cell.z = "0.0%"; // Excel % format
          }

          // ✅ Force Count column (C) to be numeric (no %)
          if (countCells[cellRef]) {
            cell.z = "0"; // integer format (no decimals, no %)
          }
        }
      }

      // Set column widths
      ws["!cols"] = [
        { wch: 30 },
        { wch: 15 },
        { wch: 10 },
      ];

      // Create workbook and export
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, "Sheet1");

      XLSX.writeFile(wb, `${question.replace(/\s+/g, "_")}.xlsx`);
    } catch (err) {
      console.error("❌ Export error:", err);
    }
  }



  function exportCrosstabToExcel(divId, matrixObj = {}, question = "Report") {
    try {
      if (!matrixObj || !matrixObj.rows) {
        console.warn("⚠️ Invalid matrixObj");
        return;
      }

      const { base, columns = [], rows = [] } = matrixObj;

      // --- Helper to build both sheets ---
      const buildSheetData = (mode) => {
        const exportData = [];
        const percentCells = {}; // track which cells need % format
        let currentRow = 0;

        exportData.push([`${question}`]);
        currentRow++;
        exportData.push([]);
        currentRow++;
        exportData.push([`Base = ${base}`]);
        currentRow++;
        exportData.push([]);
        currentRow++;
        exportData.push(["Label", ...columns.map((c) => c.trim())]);
        currentRow++;

        rows.forEach((row) => {
          const rowData = [row.label];
          row.cells.forEach((cell, colIdx) => {
            if (mode === "count") {
              rowData.push(cell.count ?? "");
            } else if (mode === "pct") {
              // ✅ Convert string percent → real Excel percentage value
              const pctVal = cell.pct ? parseFloat(cell.pct.replace("%", "")) / 100 : "";
              rowData.push(pctVal);

              // record the cell position for % formatting
              if (pctVal !== "") {
                const cellRef = XLSX.utils.encode_cell({
                  r: currentRow,
                  c: colIdx + 1, // +1 because label is column 0
                });
                percentCells[cellRef] = true;
              }
            }
          });
          exportData.push(rowData);
          currentRow++;
        });

        return { exportData, percentCells };
      };

      const { exportData: countData } = buildSheetData("count");
      const { exportData: pctData, percentCells } = buildSheetData("pct");

      // --- Helper to create formatted sheet ---
      const makeSheet = (data, columns, mode, percentCells) => {
        const ws = XLSX.utils.aoa_to_sheet(data);

        // Merge title + base
        ws["!merges"] = [
          { s: { r: 0, c: 0 }, e: { r: 0, c: columns.length } },
          { s: { r: 2, c: 0 }, e: { r: 2, c: columns.length } },
        ];

        // Styles
        const titleStyle = {
          font: { bold: true, sz: 14, color: { rgb: "1E4D2B" } },
          alignment: { horizontal: "left", vertical: "center" },
        };
        const baseStyle = {
          font: { bold: true },
          fill: { fgColor: { rgb: "E7E6E6" } },
          alignment: { horizontal: "left" },
        };
        const headerStyle = {
          font: { bold: true },
          fill: { fgColor: { rgb: "D9E1F2" } },
          alignment: { horizontal: "center" },
        };
        const totalStyle = { font: { bold: true } };

        const range = XLSX.utils.decode_range(ws["!ref"]);

        for (let R = range.s.r; R <= range.e.r; R++) {
          for (let C = range.s.c; C <= range.e.c; C++) {
            const cellRef = XLSX.utils.encode_cell({ r: R, c: C });
            if (!ws[cellRef]) continue;

            const cell = ws[cellRef];
            cell.s = cell.s || {};

            // Title
            if (R === 0) cell.s = titleStyle;
            // Base
            if (R === 2) cell.s = baseStyle;
            // Header
            if (R === 4) cell.s = headerStyle;
            // Total
            if (rows[R - 5]?.label?.toLowerCase().includes("total")) {
              cell.s = totalStyle;
            }

            // ✅ Apply % format only for percentage sheet
            if (mode === "pct" && percentCells[cellRef]) {
              cell.z = "0.0%"; // Excel format
            }
          }
        }

        ws["!cols"] = [{ wch: 25 }, ...columns.map(() => ({ wch: 15 }))];
        return ws;
      };

      // --- Create sheets
      const wsCount = makeSheet(countData, columns, "count", {});
      const wsPct = makeSheet(pctData, columns, "pct", percentCells);

      // --- Create workbook ---
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, wsCount, "Count");
      XLSX.utils.book_append_sheet(wb, wsPct, "Percentage");

      // --- Export ---
      XLSX.writeFile(wb, `${question.replace(/\s+/g, "_")}.xlsx`);
    } catch (err) {
      console.error("❌ Export error:", err);
    }
  }




  function exportMultiTopbreakToExcel(divId, data = [], question = "Report") {
    try {
      if (!Array.isArray(data) || data.length === 0) {
        console.warn("⚠️ No data to export");
        return;
      }

      const wb = XLSX.utils.book_new();

      // 🟢 Helper: Build combined data (mode = "count" or "pct")
      const buildCombinedData = (mode) => {
        const exportData = [];
        const cellMeta = {}; // % style tracking
        let currentRow = 0;

        data.forEach((item) => {
          const { topbreak_label, Matrix } = item;
          const { base, columns = [], rows = [] } = Matrix;

          // Dynamically handle column labels
          const colHeaders = columns.map((c) =>
            typeof c === "object"
              ? `${c.label}${c.letter ? " (" + c.letter + ")" : ""}`
              : c
          );

          // Section title
          exportData.push([`${question} - ${topbreak_label}`]);
          currentRow++;
          exportData.push([]);
          currentRow++;
          exportData.push([`Base = ${base}`]);
          currentRow++;
          exportData.push([]);
          currentRow++;

          // Header
          exportData.push(["Label", ...colHeaders]);
          currentRow++;

          // Rows
          rows.forEach((row) => {
            const rowData = [row.label];
            row.cells.forEach((cell, colIdx) => {
              if (mode === "count") {
                const val = cell.sig ? `${cell.count}${cell.sig}` : cell.count ?? "";
                rowData.push(val);
              } else if (mode === "pct") {
                const pctVal = cell.pct ? parseFloat(cell.pct.replace("%", "")) / 100 : "";
                rowData.push(pctVal);
                if (pctVal !== "") {
                  const cellRef = XLSX.utils.encode_cell({
                    r: currentRow,
                    c: colIdx + 1,
                  });
                  cellMeta[cellRef] = { numFmt: "0.0%" };
                }
              }
            });
            exportData.push(rowData);
            currentRow++;
          });

          // Add spacing between each topbreak
          exportData.push([]);
          exportData.push([]);
          currentRow += 2;
        });

        return { exportData, cellMeta };
      };

      const { exportData: countData } = buildCombinedData("count");
      const { exportData: pctData, cellMeta } = buildCombinedData("pct");

      // 🟢 Helper: Style each sheet
      const makeSheet = (data, mode, cellMeta) => {
        const ws = XLSX.utils.aoa_to_sheet(data);

        const titleStyle = {
          font: { bold: true, sz: 13, color: { rgb: "1E4D2B" } },
          alignment: { horizontal: "left", vertical: "center" },
        };
        const baseStyle = {
          font: { bold: true },
          fill: { fgColor: { rgb: "E7E6E6" } },
          alignment: { horizontal: "left" },
        };
        const headerStyle = {
          font: { bold: true },
          fill: { fgColor: { rgb: "D9E1F2" } },
          alignment: { horizontal: "center" },
        };
        const totalStyle = { font: { bold: true } };

        const range = XLSX.utils.decode_range(ws["!ref"]);

        for (let R = range.s.r; R <= range.e.r; R++) {
          for (let C = range.s.c; C <= range.e.c; C++) {
            const cellRef = XLSX.utils.encode_cell({ r: R, c: C });
            if (!ws[cellRef]) continue;
            const val = (ws[cellRef].v || "").toString().toLowerCase();

            if (val.startsWith(question.toLowerCase())) ws[cellRef].s = titleStyle;
            else if (val.startsWith("base =")) ws[cellRef].s = baseStyle;
            else if (val === "label") ws[cellRef].s = headerStyle;
            else if (val.includes("total")) ws[cellRef].s = totalStyle;

            // ✅ Real percentage format
            if (mode === "pct" && cellMeta[cellRef]) {
              ws[cellRef].z = cellMeta[cellRef].numFmt;
            }
          }
        }

        // Columns width (auto expand for large)
        ws["!cols"] = [{ wch: 25 }, ...Array(10).fill({ wch: 15 })];

        return ws;
      };

      // Create both sheets
      const wsCount = makeSheet(countData, "count");
      const wsPct = makeSheet(pctData, "pct", cellMeta);

      // Add to workbook
      XLSX.utils.book_append_sheet(wb, wsCount, "Count");
      XLSX.utils.book_append_sheet(wb, wsPct, "Percentage");

      // Save
      XLSX.writeFile(wb, `${question.replace(/\s+/g, "_")}with_topbreak.xlsx`);
    } catch (err) {
      console.error("❌ Export error:", err);
    }
  }

  function exportMergedTopbreakGridToExcel(divId, matrixObj = {}, question = "Report") {
    try {
      if (!matrixObj || !matrixObj.rows) {
        console.warn("⚠️ Invalid data format");
        return;
      }

      const { columns = [], rows = [] } = matrixObj;

      // --- Helper to build sheet data ---
      const buildSheetData = (mode) => {
        const exportData = [];
        const percentCells = {}; // for real % formatting
        let currentRow = 0;

        exportData.push([question]);
        currentRow++;
        exportData.push([]); // blank row
        currentRow++;

        // Header
        const header = ["Label", ...columns.map((c) => `${c.label} (${c.letter})`)];
        exportData.push(header);
        currentRow++;

        // Data rows
        rows.forEach((row) => {
          const rowData = [row.label];

          row.cells.forEach((cell, colIdx) => {
            if (mode === "count") {
              // Include sig if available
              const val = cell.sig ? `${cell.count}${cell.sig}` : cell.count ?? "";
              rowData.push(val);
            } else if (mode === "pct") {
              let pctVal = "";

              // ✅ If base row, use count instead of %
              if (row.label.toLowerCase() === "base") {
                pctVal = cell.count ?? "";
              } else if (cell.pct) {
                // convert string % → numeric fraction
                pctVal = parseFloat(cell.pct.replace("%", "")) / 100;
                const cellRef = XLSX.utils.encode_cell({ r: currentRow, c: colIdx + 1 });
                percentCells[cellRef] = true;
              }

              rowData.push(pctVal);
            }
          });

          exportData.push(rowData);
          currentRow++;
        });

        return { exportData, percentCells };
      };

      const { exportData: countData } = buildSheetData("count");
      const { exportData: pctData, percentCells } = buildSheetData("pct");

      // --- Helper to style sheet ---
      const makeSheet = (data, mode, percentCells) => {
        const ws = XLSX.utils.aoa_to_sheet(data);

        // Styles
        const titleStyle = {
          font: { bold: true, sz: 14, color: { rgb: "1E4D2B" } },
          alignment: { horizontal: "left", vertical: "center" },
        };
        const headerStyle = {
          font: { bold: true },
          fill: { fgColor: { rgb: "D9E1F2" } },
          alignment: { horizontal: "center" },
        };
        const baseStyle = {
          font: { bold: true },
          fill: { fgColor: { rgb: "E7E6E6" } },
          alignment: { horizontal: "left" },
        };
        const totalStyle = { font: { bold: true } };

        const range = XLSX.utils.decode_range(ws["!ref"]);

        for (let R = range.s.r; R <= range.e.r; R++) {
          for (let C = range.s.c; C <= range.e.c; C++) {
            const cellRef = XLSX.utils.encode_cell({ r: R, c: C });
            if (!ws[cellRef]) continue;
            const val = (ws[cellRef].v || "").toString().toLowerCase();

            // Apply styles
            if (R === 0) ws[cellRef].s = titleStyle;
            if (R === 2) ws[cellRef].s = headerStyle;
            if (val === "base") ws[cellRef].s = baseStyle;
            if (val.includes("total (")) ws[cellRef].s = headerStyle;

            // ✅ Apply percentage format
            if (mode === "pct" && percentCells[cellRef]) {
              ws[cellRef].z = "0.0%"; // shows 1 → 100.0%
            }
          }
        }

        // Column widths
        ws["!cols"] = [{ wch: 25 }, ...columns.map(() => ({ wch: 15 }))];

        return ws;
      };

      const wsCount = makeSheet(countData, "count");
      const wsPct = makeSheet(pctData, "pct", percentCells);

      // --- Create Workbook
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, wsCount, "Count");
      XLSX.utils.book_append_sheet(wb, wsPct, "Percentage");

      // --- Save file
      XLSX.writeFile(wb, `${question.replace(/\s+/g, "_")}with_topbreak.xlsx`);
    } catch (err) {
      console.error("❌ Export error:", err);
    }
  }


  function exportNumericGridToExcel(divId, numericData = {}, question = "Numeric Grid") {
    try {
      const matrix = numericData;
      if (!matrix || !matrix.subtables || matrix.subtables.length === 0) {
        console.warn("⚠️ No valid numeric grid data found");
        return;
      }

      const { columns = [], subtables = [] } = matrix;
      const exportData = [];
      let currentRow = 0;

      // 🟩 Loop each subtable
      subtables.forEach((sub, subIdx) => {
        const title = sub.title?.trim() || `Subtable ${subIdx + 1}`;

        // Section title
        exportData.push([`${question}: ${title}`]);
        currentRow++;
        exportData.push([]);
        currentRow++;

        // Header row
        const header = ["Label", ...columns.map((c) => `${c.label} (${c.letter})`)];
        exportData.push(header);
        currentRow++;

        // Rows
        sub.rows.forEach((row) => {
          const rowData = [row.label];

          row.values.forEach((val, idx) => {
            if (typeof val === "object" && val.count !== undefined) {
              // Base row
              rowData.push(val.count);
            } else {
              // Mean / Std Dev rows
              const sig = row.sig?.[idx] ? row.sig[idx] : "";
              const displayValue = sig && sig.trim() !== "" ? `${val}${sig}` : val ?? "";
              rowData.push(displayValue);
            }
          });

          exportData.push(rowData);
          currentRow++;
        });

        // Blank rows between tables
        exportData.push([]);
        exportData.push([]);
        currentRow += 2;
      });

      // 🟩 Create worksheet
      const ws = XLSX.utils.aoa_to_sheet(exportData);

      // 🟩 Styling
      const titleStyle = {
        font: { bold: true, sz: 14, color: { rgb: "1E4D2B" } },
        alignment: { horizontal: "left", vertical: "center" },
      };
      const headerStyle = {
        font: { bold: true },
        fill: { fgColor: { rgb: "D9E1F2" } },
        alignment: { horizontal: "center" },
      };
      const baseStyle = {
        font: { bold: true },
        fill: { fgColor: { rgb: "E7E6E6" } },
        alignment: { horizontal: "left" },
      };
      const meanStyle = {
        font: { color: { rgb: "000080" }, italic: true },
      };
      const stdDevStyle = {
        font: { color: { rgb: "555555" } },
      };

      // 🟩 Apply styles row by row
      const range = XLSX.utils.decode_range(ws["!ref"]);

      for (let R = range.s.r; R <= range.e.r; R++) {
        for (let C = range.s.c; C <= range.e.c; C++) {
          const cellRef = XLSX.utils.encode_cell({ r: R, c: C });
          if (!ws[cellRef]) continue;
          const val = (ws[cellRef].v || "").toString().toLowerCase();

          // Apply per-row styles
          if (val.startsWith(question.toLowerCase())) ws[cellRef].s = titleStyle;
          else if (val === "label") ws[cellRef].s = headerStyle;
          else if (val === "base") ws[cellRef].s = baseStyle;
          else if (val === "mean") ws[cellRef].s = meanStyle;
          else if (val.includes("std dev")) ws[cellRef].s = stdDevStyle;
        }
      }

      // 🟩 Adjust column width
      ws["!cols"] = [{ wch: 25 }, ...columns.map(() => ({ wch: 20 }))];

      // 🟩 Workbook
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, "Numeric Grid");

      XLSX.writeFile(wb, `${question.replace(/\s+/g, "_")}_Combined_NumericGrid.xlsx`);
    } catch (err) {
      console.error("❌ Export error:", err);
    }
  }





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

    // --- keep header rows (if any)
    const fixedTop = rows.filter((r) => r.label === "");

    // --- identify logical groups
    const totalRow = rows.filter((r) => r.label === "Total");
    const summaryRows = rows.filter((r) =>
      ["Top 2", "Top 3", "Bottom 2", "Bottom 3", "Mean", "StdDev", "Count"].includes(r.label)
    );

    // --- normal data rows (attributes only)
    let sortable = rows.filter(
      (r) =>
        !["", "Total", "Top 2", "Top 3", "Bottom 2", "Bottom 3", "Mean", "StdDev", "Count"].includes(
          r.label
        )
    );

    // --- sort attributes if column sorting active
    if (cfg.colIndex !== null && cfg.direction !== "none") {
      sortable = sortable.sort((a, b) => {
        const aVal = a.cells?.[cfg.colIndex]?.count || 0;
        const bVal = b.cells?.[cfg.colIndex]?.count || 0;
        return cfg.direction === "desc" ? bVal - aVal : aVal - bVal;
      });
    }

    // --- Final order: top → attributes → total → summaries
    return [...fixedTop, ...sortable, ...totalRow, ...summaryRows];
  };


  // -------- Word Cloud Renderer ----------
  const renderWordCloud = ({ data, question }) => {
    useEffect(() => {
      if (!data || data.length === 0) return;

      const chartDom = document.getElementById(`wordcloud_${question}`);
      const chart = echarts.init(chartDom);

      const option = {
        title: {
          text: question || "Word Cloud",
          left: "center",
          textStyle: {
            fontSize: 18,
            color: "#2563eb", // blue
            fontWeight: "bold",
          },
        },
        tooltip: {
          show: true,
          formatter: (item) => `${item.name}: ${item.value}`,
        },
        series: [
          {
            type: "wordCloud",
            shape: "circle", // or 'diamond' | 'triangle' | 'pentagon' | 'star'
            keepAspect: false,
            gridSize: 2,
            sizeRange: [12, 60], // min/max font size
            rotationRange: [-45, 90],
            rotationStep: 45,
            textStyle: {
              fontFamily: "Poppins, sans-serif",
              fontWeight: "bold",
              color: () => {
                // Random bright color
                const r = Math.round(Math.random() * 160);
                const g = Math.round(Math.random() * 160);
                const b = Math.round(Math.random() * 160);
                return `rgb(${r}, ${g}, ${b})`;
              },
            },
            emphasis: {
              textStyle: {
                shadowBlur: 10,
                shadowColor: "#333",
                color: "#000",
              },
            },
            data: data,
          },
        ],
      };

      chart.setOption(option);
      window.addEventListener("resize", chart.resize);

      return () => {
        window.removeEventListener("resize", chart.resize);
        chart.dispose();
      };
    }, [data, question]);

    return (
      <div
        id={`wordcloud_${question}`}
        style={{
          width: "100%",
          height: "500px",
          margin: "0 auto",
          background: "#fff",
          borderRadius: "12px",
          boxShadow: "0 0 8px rgba(0,0,0,0.1)",
          padding: "20px",
        }}
      ></div>
    );
  };


  // -------- Numeric Grid Renderer ----------
  const renderNumericGrid = (matrixObj, groupIdx, question) => {
    if (!matrixObj) return null;

    const { base, columns, rows } = matrixObj.Matrix;

    return (
      <div
        id={`matrix_${groupIdx}`}
        className="bg-white border rounded-lg shadow p-4 mb-6"
      >
        {/* Header: question + export button */}
        <div className="flex justify-between items-center mb-3">
          <div className="text-green-700 font-semibold">{question}</div>
          <button className="text-sm text-blue-600 hover:underline"
            onClick={() => exportNumericGridToExcel(groupIdx, matrixObj, question)}>
            Export
          </button>
        </div>

        {/* Table */}
        <table className="w-full text-sm border-collapse border border-black-300">
          <thead className="bg-gray-300">
            <tr>
              <th className="border border-black-300 p-2 text-left">Base</th>
              <th className="border border-black-300 p-2 text-center" colSpan={columns.length}>
                <div className="flex flex-col items-center">
                  {/* <span className="font-semibold text-gray-800">
                    {base.avg_mean}
                  </span> */}
                  <span className="text-xs text-gray-600">
                    {base.count}
                  </span>
                </div>
              </th>

            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <React.Fragment key={i}>
                <tr>
                  <td className="border border-black-300 p-2  text-center font-normal" rowSpan={columns.length}>{row.label}</td>
                  <td className="border border-black-300 p-2">{columns[0]}</td>
                  <td className="border border-black-300 p-2 text-left">
                    {row.values[0]}
                  </td>
                </tr>
                <tr>
                  <td className="border border-black-300 p-2">{columns[1]}</td>
                  <td className="border border-black-300 p-2 text-left">
                    {row.values[1]}
                  </td>
                </tr>
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div >
    );
  };
  // -------- Numeric Grid with Topbreak Renderer ----------
  const renderNumericGridMultiTable = (matrixObj, groupIdx, question) => {
    const { columns, subtables } = matrixObj;

    return (
      <div
        id={groupIdx} className="bg-white border rounded-lg shadow p-4 mb-6">
        <div className="flex justify-between items-center mb-3">
          <div className="text-green-700 font-semibold">{question}</div>
          <button className="text-sm text-blue-600 hover:underline"
            onClick={() => exportNumericGridToExcel(groupIdx, matrixObj, question)}>Export</button>
        </div>

        {subtables.map((table, idx) => (
          <div key={idx} className="mb-4 border rounded-lg overflow-hidden">
            <div className="bg-gray-50 text-blue-600 font-bold mb-2">{table.title}</div>
            <table className="w-full text-sm border-collapse border border-gray-300">
              <thead className="bg-gray-100">
                <tr>
                  {/*  */}
                  <th className="border border-gray-300 p-2 text-left"></th>
                  {columns.map((col, i) => (
                    <th key={i} className="border border-gray-300 p-2 text-center">
                      {col.label} <span className="text-blue-600 font-bold mr-1">{col.letter}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, i) => (
                  <tr key={i}>
                    <td className="border border-gray-300 p-2 font-medium">{row.label}</td>
                    {row.values.map((val, j) => (
                      <td key={j} className="border border-gray-300 p-2 text-center">
                        {typeof val === "object" ? (
                          <>
                            <span>{val.pct}</span><br />
                            <span>{val.count}</span>
                          </>
                        ) : (
                          <>
                            {val}
                            {row.sig && row.sig[j] && (
                              <sup className="text-red-600 ml-1">{row.sig[j]}</sup>
                            )}
                          </>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    );
  };
  // -------- NPS Renderer ----------


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
          <button className="text-sm text-blue-600 hover:underline"
            onClick={() => exportToExcel(groupIdx, matrixObj, question)}>Export</button>
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
            <button className="text-sm text-blue-600 hover:underline"
              onClick={() => exportMultiTopbreakToExcel(groupIdx, matrixObj, question)}>Export</button>
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
                        <th className="border px-3 py-2 text-left bg-gray-100"></th>
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
          <button className="text-sm text-blue-600 hover:underline"
            onClick={() => exportMergedTopbreakGridToExcel(groupIdx, matrixObj, question)}>Export</button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-xs">
            <thead className="bg-gray-100">
              <tr>
                <th className="border px-3 py-2 text-left bg-gray-100"></th>
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

    const matrix = qcObj.Matrix;
    // 🔹 Case 1: SRG in Quick Crosstab
    if (type === "SRG" || type === "MRG" && qcObj.Matrix) {
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
            <button className="text-sm text-blue-600 hover:underline"
              onClick={() => exportCrosstabToExcel(groupIdx, matrix, question)}>Export</button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse text-xs">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border px-3 py-2 text-left bg-gray-100"></th>
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
            <button className="text-sm text-blue-600 hover:underline"
              onClick={() => exportToExcel(groupIdx, rows, question)}>Export</button>
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
                <th className="border px-3 py-2 text-left bg-gray-100"></th>
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
    const [searchTerm, setSearchTerm] = useState("");

    return (
      <div className="fixed top-0 left-0 h-full w-64 bg-gray-100 border-r shadow-lg z-40 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b">
          <h2 className="font-bold text-gray-700">Questions</h2>
        </div>

        {/* Search box */}
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
                id={`sidebar-item-${idx}`}
                key={idx}
                className={`px-3 py-2 cursor-pointer rounded text-sm ${activeIndex === idx
                  ? "bg-blue-200 font-bold"
                  : "hover:bg-blue-100"
                  }`}
                onClick={() => {
                  const el = document.getElementById(`matrix_${idx}`);
                  if (el) {
                    const top = el.offsetTop;

                    // ✅ Auto switch between smooth and jump
                    const behavior = groups.length > 100 ? "auto" : "auto";

                    window.scrollTo({
                      top,
                      behavior,
                    });
                  }

                  // ✅ Ensure sidebar item stays visible
                  document
                    .getElementById(`sidebar-item-${idx}`)
                    ?.scrollIntoView({ block: "nearest", behavior: "auto" });
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
            else if ((grp.type !== "SRG" || grp.type !== "MRG") && grp.type === "NRG" && grp.crosstab_type === "new") {

              card = renderNumericGridMultiTable(grp.data.matrix, `matrix_${gIdx}`, grp.question);
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
            else if (grp.type === "WordCloud") {
              if (grp.data.length !== 0) {
                return <WordCloudChart data={grp.data} chartId={`${gIdx}`} question={grp.question} />;
              }
            }
            else if (grp.type === "NRG") {
              return renderNumericGrid(grp.data.matrix[0], gIdx, grp.question);
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