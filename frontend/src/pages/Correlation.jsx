import React, { useEffect, useState } from "react";
import API from "../services/api";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function Correlation() {
  const [cars, setCars] = useState([]);

  useEffect(() => {
    API.get("/sample?n=100").then((res) => setCars(res.data));
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
