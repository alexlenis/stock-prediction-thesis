import React from "react";
export default function SignalsPro({ data }) {
  if (!data) return <p>Loading...</p>;
  return (
    <div>
      <h2 className="text-lg font-bold mb-4">Signals</h2>
      <div className="space-y-2">
        <div>RF: {data.rf}</div>
        <div>XGB: {data.xgb}</div>
        <div>LGBM: {data.lgbm}</div>
        <div>LOG: {data.logistic}</div>
        <div>LSTM: {data.lstm}</div>
      </div>
    </div>
  );
}