export default function FinalPro({ data }) {
  if (!data) return null;

  return (
    <div className="mt-6 p-6 text-center text-2xl font-bold rounded-2xl final-glow">
      🚀 FINAL SIGNAL: {data.final}
    </div>
  );
}