import React, { useState } from "react";

function SortableTable({ columns, rows }) {
  const [sortConfig, setSortConfig] = useState({
    colIndex: null,
    direction: "none",
  });

  const handleSort = (colIdx) => {
    setSortConfig((prev) => {
      if (prev.colIndex === colIdx) {
        if (prev.direction === "desc")
          return { colIndex: colIdx, direction: "asc" };
        if (prev.direction === "asc")
          return { colIndex: colIdx, direction: "none" }; // reset to meta order
        return { colIndex: colIdx, direction: "desc" };
      }
      return { colIndex: colIdx, direction: "desc" };
    });
  };

  // --- Apply sorting ---
  let displayRows = [...rows];

  const fixedTop = displayRows.filter((r) => r.label === "Total");
  const fixedBottom = displayRows.filter((r) =>
    ["Mean", "Median", "StdDev"].includes(r.label)
  );
  let sortableRows = displayRows.filter(
    (r) => !["Total", "Mean", "Median", "StdDev"].includes(r.label)
  );

  if (sortConfig.colIndex !== null && sortConfig.direction !== "none") {
    sortableRows = sortableRows.sort((a, b) => {
      const valA = a.cells?.[sortConfig.colIndex]?.count || 0;
      const valB = b.cells?.[sortConfig.colIndex]?.count || 0;
      return sortConfig.direction === "desc" ? valB - valA : valA - valB;
    });
  }

  displayRows = [...fixedTop, ...sortableRows, ...fixedBottom];

  return (
    <table className="min-w-full border text-xs">
      <thead className="bg-gray-200">
        <tr>
          <th className="border px-2 py-1 text-left">Scale</th>
          {columns.map((col, cIdx) => (
            <th
              key={cIdx}
              onClick={() => handleSort(cIdx)}
              className={`border px-2 py-1 text-center cursor-pointer hover:bg-gray-300
                ${
                  sortConfig.colIndex === cIdx &&
                  sortConfig.direction !== "none"
                    ? "text-blue-600 font-bold"
                    : "text-gray-800"
                }`}
            >
              {col}
              {sortConfig.colIndex === cIdx && (
                <span className="ml-1 text-xs">
                  {sortConfig.direction === "desc"
                    ? "↓"
                    : sortConfig.direction === "asc"
                    ? "↑"
                    : "↺"}
                </span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {displayRows.map((row, rIdx) => (
          <tr key={rIdx} className={rIdx % 2 === 0 ? "" : ""}>
            <td className="border px-2 py-1 font-semibold">{row.label}</td>
            {row.cells.map((cell, idx) => (
              <td key={idx} className="border px-2 py-1 text-center">
                {cell.pct ? (
                  <div className="flex flex-col items-center">
                    <span>{cell.pct}</span>
                    <span>{cell.count}</span>
                  </div>
                ) : (
                  <span>{cell.count}</span>
                )}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default SortableTable;
