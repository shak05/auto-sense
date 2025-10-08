/*



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
  const brandCounts = eda?.brand_counts
  ? Object.entries(eda.brand_counts).map(([brand, count]) => ({ brand, count }))
  : [];


  // --- NEW LOGIC TO EXTRACT CORRELATIONS WITH PRICE ---
  const priceCorrelations = [];
  if (eda?.correlations?.price_in_lakhs) {
    const corrMap = eda.correlations.price_in_lakhs;
    // Keys to display: only show the features, not the price itself
    const keys = ['age', 'kms_driven', 'mileagekmpl', 'enginecc', 'max_powerbhp', 'torqueNm'];
    
    // Create an array of {feature: value} for display
    keys.forEach(key => {
        if (corrMap[key] !== undefined) {
            priceCorrelations.push({
                feature: key, 
                value: corrMap[key]
            });
        }
    });
    
    // Sort by absolute value to show the strongest relationships first
    priceCorrelations.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }
  // ----------------------------------------------------

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
          <h3>Pearson Correlation with Price</h3> 
          {priceCorrelations.length > 0 ? (
            <div className="correlation-list"> 
              {priceCorrelations.map((item) => {
                const isNegative = item.value < 0;
                const sign = isNegative ? '⬇️' : '⬆️'; // Emoji indicator for direction
                const color = isNegative ? '#ff6b6b' : '#3c79a1'; // Red for negative, Blue for positive

                return (
                  <div key={item.feature} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                    <strong>{item.feature.replace(/_/g, ' ').toUpperCase()}</strong>
                    <span style={{ color: color, fontWeight: 'bold' }}>
                        {sign} {item.value.toFixed(3)}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p>Loading correlations or no data found...</p>
          )}
        </div>
      </div>
    </div>
  );
}

*/

import React, { useEffect, useState } from "react";
import API from "../services/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function Summary() {
  const [summary, setSummary] = useState(null);
  const [eda, setEda] = useState(null);

  useEffect(() => {
    API.get("/summary")
      .then((r) => setSummary(r.data))
      .catch(() => {});
    API.get("/eda")
      .then((r) => setEda(r.data))
      .catch(() => {});
  }, []);

  const brandCounts = eda?.brand_counts
    ? Object.entries(eda.brand_counts).map(([k, v]) => ({ brand: k, count: v }))
    : [];

  return (
    <div>
      {/* Dataset Summary */}
      <div className="panel">
        <h2>Dataset Summary</h2>
        {summary ? (
          <div>
            <p>
              <strong>Rows:</strong> {summary.rows} &nbsp; <strong>Columns:</strong> {summary.columns}
            </p>
           {/*<p className="smallmuted">
              Model metrics: price test score <strong>{summary.price_test_score?.toFixed(3) ?? "N/A"}</strong>
        </p>*/}
          </div>
        ) : (
          <p>Loading summary...</p>
        )}
      </div>

      {/* Top Brands */}
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
      </div>
    </div>
  );
}

