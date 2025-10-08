/*import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="hero">
      <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        Car Price Market Risk Analytics
      </motion.h1>
      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
        Analyze car values, risks & correlations with smooth interactive dashboards
      </motion.p>
      <motion.button 
        onClick={() => navigate("/market")} 
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}>
      Explore Market Value
      </motion.button>
    </div>
  );
}
*/

import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="hero flex flex-col justify-center items-center h-screen px-6 bg-gradient-to-br from-teal-600 to-teal-400 text-center">
      
      {/* Heading */}
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-5xl font-bold text-white mb-6"
      >
        AutoSense
      </motion.h1>

      {/* Paragraph */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-white text-lg max-w-2xl mb-10"
      >
        Analyze car values, risks & correlations with smooth interactive dashboards
      </motion.p>

      {/* Button */}
      <motion.button
        onClick={() => navigate("/market")}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white text-teal-600 font-semibold py-3 px-6 rounded-lg shadow-lg hover:bg-gray-100 transition"
      >
        Explore Market Value
      </motion.button>
    </div>
  );
}
