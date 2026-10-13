import { Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import About from "./pages/About";
import Dashboard from "./pages/Dashboard";
import Dataset from "./pages/Dataset";
import Model from "./pages/Model";

export default function App() {
  return (
    <div className="min-h-screen bg-surface-light dark:bg-surface-dark">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/model" element={<Model />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}
