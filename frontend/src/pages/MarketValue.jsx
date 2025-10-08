

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from "framer-motion";
import API from "../services/api";

export default function MarketValue() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [carNames, setCarNames] = useState([]);
  const [selectedCarName, setSelectedCarName] = useState('');
  const [loadingCars, setLoadingCars] = useState(true);

  const fuelTypes = ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Unknown"];
  const transmissions = ["Manual", "Automatic", "Unknown"];
  const ownerships = ["First Owner", "Second Owner", "Third Owner", "Fourth Owner", "Unknown"];

  // Fetch available car names
  useEffect(() => {
    const fetchCarNames = async () => {
      try {
        const res = await API.get("/car_names");
        setCarNames(res.data.car_names);
        if (res.data.car_names.length > 0) setSelectedCarName(res.data.car_names[0]);
      } catch (error) {
        console.error("Failed to fetch car names:", error);
      } finally {
        setLoadingCars(false);
      }
    };
    fetchCarNames();
  }, []);

  // Handle form submit
  const handleSelect = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const details = Object.fromEntries(formData.entries());

    const carDetailsPayload = {
      ...details,
      manufacturing_year: parseInt(details.manufacturing_year),
      kms_driven: parseInt(details.kms_driven),
      mileagekmpl: parseFloat(details.mileagekmpl),
    };

    try {
      const res = await API.post("/predict_risk", carDetailsPayload);
      setPredictionResult(res.data);
    } catch (error) {
      console.error("Prediction failed:", error.response?.data || error);
      alert("Prediction failed. Check console for details.");
      setPredictionResult({ predicted_market_risk: "Error", predicted_price_in_lakhs: 0 });
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="min-h-screen bg-gradient-to-br from-teal-600 to-teal-400 flex flex-col items-center py-10 px-6"
    >
      <motion.h2
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="text-white text-4xl font-bold mb-10 tracking-wide text-center"
      >
        Car Market Value & Risk Predictor
      </motion.h2>

      <div className="grid md:grid-cols-2 gap-8 max-w-5xl w-full">
        
        <motion.div
          initial={{ x: -30, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="bg-white rounded-2xl shadow-xl p-8 space-y-4"
        >
          <h3 className="text-2xl font-semibold text-gray-800 mb-4">Car Details</h3>
          <form onSubmit={handleSelect} className="grid grid-cols-2 gap-4">
            
            <label className="col-span-2">
              <span className="text-sm font-semibold text-gray-600">Car Name</span>
              {loadingCars ? (
                <select className="input" disabled><option>Loading...</option></select>
              ) : (
                <select
                  name="car_name"
                  value={selectedCarName}
                  onChange={(e) => setSelectedCarName(e.target.value)}
                  required
                  className="input"
                >
                  <option value="" disabled>-- Select Car --</option>
                  {carNames.map(name => <option key={name} value={name}>{name}</option>)}
                </select>
              )}
            </label>

            <label>
              <span>Year Built</span>
              <input name="manufacturing_year" type="number" defaultValue={2018} className="input" required />
            </label>

            <label>
              <span>Kms Driven</span>
              <input name="kms_driven" type="number" defaultValue={50000} className="input" required />
            </label>

            <label>
              <span>Fuel Type</span>
              <select name="fuel_type" className="input">
                {fuelTypes.map(f => <option key={f}>{f}</option>)}
              </select>
            </label>

            <label>
              <span>Transmission</span>
              <select name="transmission" className="input">
                {transmissions.map(t => <option key={t}>{t}</option>)}
              </select>
            </label>

            <label className="col-span-2">
              <span>Ownership</span>
              <select name="ownership" className="input">
                {ownerships.map(o => <option key={o}>{o}</option>)}
              </select>
            </label>

            <label className="col-span-2">
              <span>Mileage (kmpl)</span>
              <input name="mileagekmpl" type="number" step="0.1" defaultValue={18.0} className="input" />
            </label>

            <button
              type="submit"
              className="col-span-2 bg-teal-600 hover:bg-teal-700 text-white font-semibold py-2.5 rounded-lg transition-all duration-300 shadow-md hover:shadow-lg"
            >
              Analyze Market Risk
            </button>
          </form>
        </motion.div>

      
        <motion.div
          initial={{ x: 30, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="bg-white rounded-2xl shadow-xl p-8 flex flex-col justify-center items-center text-center"
        >
          <h3 className="text-2xl font-semibold text-gray-800 mb-6">Prediction Result</h3>
          {predictionResult ? (
            <AnimatePresence>
              <motion.div
                key={predictionResult.predicted_market_risk}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.3 }}
                className="space-y-4"
              >
                <p className="text-lg font-semibold text-gray-700">
                  Market Risk:{" "}
                  <span
                    className={`font-bold ${
                      predictionResult.predicted_market_risk === "High"
                        ? "text-red-500"
                        : predictionResult.predicted_market_risk === "Medium"
                        ? "text-yellow-500"
                        : "text-green-500"
                    }`}
                  >
                    {predictionResult.predicted_market_risk}
                  </span>
                </p>

                <p className="text-lg font-semibold text-gray-700">
                  Predicted Price:{" "}
                  <span className="text-teal-600 font-bold">
                    {predictionResult.predicted_price_in_lakhs} Lakh
                  </span>
                </p>

                <p className="text-gray-600 italic">
                  This prediction reflects both market trends and ownership impact.
                </p>
              </motion.div>
            </AnimatePresence>
          ) : (
            <p className="text-gray-500">
              Fill in car details and click <strong>Analyze</strong> to see the results.
            </p>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}

