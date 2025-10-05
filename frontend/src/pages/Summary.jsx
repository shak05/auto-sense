import React, { useEffect, useState } from "react";
import API from "../services/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function Summary() {
  const [summary, setSummary] = useState(null);
  const [eda, setEda] = useState(null);

  useEffect(() => {
    API.get("/summary").then((r) => setSummary(r.data)).catch(() => {});
    API.get("/eda").then((r) => setEda(r.data)).catch(() => {});
  }, []);

  const brandCounts = eda?.brand_counts ? Object.entries(eda.brand_counts).map(([k, v]) => ({ brand: k, count: v })) : [];

  return (
    <div>
      <div className="panel">
        <h2>Dataset Summary</h2>
        {summary ? (
          <div>
            <p>
              <strong>Rows:</strong> {summary.rows} &nbsp; <strong>Columns:</strong> {summary.columns}
            </p>
            <p className="smallmuted">
              Model metrics: price test score <strong>{summary.price_test_score?.toFixed(3) ?? "N/A"}</strong>
            </p>
          </div>
        ) : (
          <p>Loading summary...</p>
        )}
      </div>

      <div className="grid">
        <div className="panel">
          <h3>Top Brands</h3>
          {brandCounts.length === 0 ? (
            <p>Loading...</p>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={brandCounts}>
                <XAxis dataKey="brand" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#00d1ff" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="panel">
          <h3>Correlations (sample)</h3>
          {eda ? (
            <div className="smallmuted">
              {Object.entries(eda.correlations || {}).slice(0, 6).map(([k, v]) => (
                <div key={k}>
                  <strong>{k}</strong>: {Number(v).toFixed(3)}
                </div>
              ))}
            </div>
          ) : (
            <p>Loading EDA...</p>
          )}
        </div>
      </div>
    </div>
  );
}
