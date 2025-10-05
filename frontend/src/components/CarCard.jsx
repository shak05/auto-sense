import React from "react";

export default function CarCard({ item, onClick, selected }) {
  return (
    <div
      className="car-card"
      onClick={() => onClick(item)}
      style={{
        border: selected ? "2px solid rgba(0,209,255,0.2)" : "2px solid transparent",
        cursor: "pointer",
      }}
    >
      <div style={{ height: 110, borderRadius: 8, overflow: "hidden", background: "rgba(255,255,255,0.02)" }}>
        <img
          src={
            item._thumbnail ||
            "https://source.unsplash.com/collection/189093/400x300?sig=" + (item.car_name?.length || Math.random())
          }
          alt={item.car_name}
        />
      </div>

      <h4>{item.car_name}</h4>
      <div className="smallmuted">Year: {item.manufacturing_year ?? item.registration_year}</div>
      <div className="smallmuted">Price: ₹ {item.price_in_lakhs ?? "—"}L</div>
    </div>
  );
}
