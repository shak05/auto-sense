/*import React, { useEffect, useState } from "react";
import API from "../services/api";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function Correlation() {
  const [cars, setCars] = useState([]);

// --- FIXED CODE ---
useEffect(() => {
  // Use the existing /clusters endpoint which returns a 'sample' list
  API.get("/clusters").then((res) => setCars(res.data.sample));
  }, []);

  const scatterData = cars.map((c) => ({
    x: c.kms_driven,
    y: c.price_in_lakhs,
    name: c.car_name,
  }));

  return (
    <div className="panel">
      <h2>Correlation Analysis (Kms vs Price)</h2>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <CartesianGrid />
          <XAxis type="number" dataKey="x" name="Kms Driven" />
          <YAxis type="number" dataKey="y" name="Price" />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter data={scatterData} fill="#00d1ff" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
*/

import API from "../services/api";

import React, { useEffect, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

export default function Correlation() {
  const [cars, setCars] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    API.get("/clusters")
      .then((res) => {
        setCars(res.data.sample);
        setLoading(false);
      })
      .catch((err) => {
        setError("Failed to load data");
        setLoading(false);
      });
  }, []);

  const scatterData = cars.map((c) => ({
    x: Math.round(c.kms_driven),
    y: Number(c.price_in_lakhs.toFixed(1)),
    name: c.car_name,
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload[0]) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-900 border border-cyan-500 rounded-lg p-3 shadow-lg">
          <p className="text-white font-semibold text-sm">{data.name}</p>
          <p className="text-cyan-400 text-sm">
            Kms: {data.x.toLocaleString()}
          </p>
          <p className="text-emerald-400 text-sm">₹{data.y}L</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-8">
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl shadow-2xl border border-cyan-500/30 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent mb-2">
            Correlation Analysis
          </h1>
          <p className="text-slate-400 text-lg">
            Relationship between Distance Driven & Market Price
          </p>
          <div className="mt-4 flex gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-cyan-400" />
              <span className="text-slate-300">Distance Driven (Kms)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-400" />
              <span className="text-slate-300">Price (Lakhs ₹)</span>
            </div>
          </div>
        </div>

        {/* Chart Container */}
        <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-6 backdrop-blur-sm">
          {loading ? (
            <div className="h-96 flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin mx-auto mb-4" />
                <p className="text-slate-400">Loading data...</p>
              </div>
            </div>
          ) : error ? (
            <div className="h-96 flex items-center justify-center">
              <p className="text-red-400">{error}</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(71, 85, 105, 0.3)"
                  vertical={false}
                />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Kms Driven"
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(71, 85, 105, 0.5)" }}
                  label={{
                    value: "Distance Driven (Kms)",
                    position: "insideBottomRight",
                    offset: -10,
                    fill: "#cbd5e1",
                    fontSize: 13,
                    fontWeight: 500,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Price"
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(71, 85, 105, 0.5)" }}
                  label={{
                    value: "Price (Lakhs ₹)",
                    angle: -90,
                    position: "insideLeft",
                    fill: "#cbd5e1",
                    fontSize: 13,
                    fontWeight: 500,
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Scatter
                  data={scatterData}
                  fill="#06b6d4"
                  fillOpacity={0.7}
                  name="Cars"
                />
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Stats Footer */}
        <div className="mt-6 grid grid-cols-3 gap-4">
          <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600/50">
            <p className="text-slate-400 text-sm mb-1">Total Samples</p>
            <p className="text-2xl font-bold text-cyan-400">{cars.length}</p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600/50">
            <p className="text-slate-400 text-sm mb-1">Avg Kms</p>
            <p className="text-2xl font-bold text-cyan-400">
              {cars.length > 0
                ? (
                    cars.reduce((sum, c) => sum + c.kms_driven, 0) / cars.length
                  )
                    .toLocaleString()
                    .substring(0, 6)
                : "—"}
            </p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600/50">
            <p className="text-slate-400 text-sm mb-1">Avg Price</p>
            <p className="text-2xl font-bold text-emerald-400">
              {cars.length > 0
                ? (
                    cars.reduce((sum, c) => sum + c.price_in_lakhs, 0) /
                    cars.length
                  ).toFixed(1)
                : "—"}
              L
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
