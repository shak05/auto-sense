// src/App.jsx
import React from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import Home from "./pages/Home";
import Summary from "./pages/Summary";
import MarketValue from "./pages/MarketValue";
import Correlation from "./pages/Correlation";
import "./index.css";

export default function App() {
  return (
    <div className="app-container">
      {/* Transparent navbar */}
      <nav className="navbar glass">
        <div className="logo">🚗 CarPriceLab</div>
        <div className="navlinks">
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/summary">Summary</NavLink>
          <NavLink to="/market">Market Value</NavLink>
          <NavLink to="/correlation">Correlation</NavLink>
        </div>
      </nav>

      {/* Animated page transitions */}
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<PageWrapper><Home/></PageWrapper>} />
          <Route path="/summary" element={<PageWrapper><Summary/></PageWrapper>} />
          <Route path="/market" element={<PageWrapper><MarketValue/></PageWrapper>} />
          <Route path="/correlation" element={<PageWrapper><Correlation/></PageWrapper>} />
        </Routes>
      </AnimatePresence>
    </div>
  );
}

// Motion wrapper for smooth slide transitions
function PageWrapper({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 80 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -80 }}
      transition={{ duration: 0.6 }}
      style={{ minHeight: "100vh", padding: "40px 20px" }}
    >
      {children}
    </motion.div>
  );
}
