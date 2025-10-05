import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import API from "../services/api";

export default function MarketValue() {
  const [car, setCar] = useState(null);
  const [plots, setPlots] = useState(null);

  const handleSelect = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const carDetails = Object.fromEntries(formData.entries());
    setCar(carDetails);

    // fetch cluster risk from backend
    const res = await API.post("/predict_risk", carDetails);
    setPlots(res.data); 
  };

  return (
    <motion.div initial={{ x: "100%" }} animate={{ x: 0 }} transition={{ duration: 0.6 }}>
      <h2>Market Value Risk Dashboard</h2>

      <form onSubmit={handleSelect} className="panel">
        <label>Brand: <input name="brand" required /></label>
        <label>Model: <input name="car_name" required /></label>
        <label>Mileage: <input name="mileagekmpl" type="number" /></label>
        <label>Age: <input name="age" type="number" /></label>
        <label>Price (Lakhs): <input name="price_in_lakhs" type="number" /></label>
        <button type="submit">Analyze</button>
      </form>

      <AnimatePresence>
        {plots && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <h3>📊 Market Risk Category: {plots.predicted_market_risk}</h3>
            <div style={{ marginTop: "20px" }}>
              {/* Here insert Scatter plot / Box plot components */}
              <p>Scatter & Box plots will render here (from dataset correlations)</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
