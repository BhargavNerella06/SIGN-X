import { NavLink } from "react-router-dom";
import { useEffect, useState } from "react";

const LINKS = [
  { to: "/", label: "Dashboard" },
  { to: "/dataset", label: "Dataset" },
  { to: "/model", label: "Model" },
  { to: "/about", label: "About" },
];

function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      const stored = localStorage.getItem("signx-theme");
      if (stored) return stored === "dark";
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    } catch {
      return false;
    }
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("signx-theme", dark ? "dark" : "light");
    } catch {
      /* private browsing / storage disabled — theme just won't persist */
    }
  }, [dark]);

  return { dark, toggle: () => setDark((d) => !d) };
}

export default function Navbar() {
  const { dark, toggle } = useTheme();

  return (
    <header className="border-b border-slate-200 bg-panel-light/80 backdrop-blur dark:border-slate-800 dark:bg-panel-dark/80">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-2">
          <span className="text-xl font-black tracking-tight text-brand-600 dark:text-brand-400">SIGN-X</span>
          <span className="hidden text-xs text-slate-400 sm:inline">ISL → Text → Voice</span>
        </div>

        <nav className="flex items-center gap-1">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                `rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-brand-600 text-white"
                    : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
          <button
            onClick={toggle}
            aria-label="Toggle dark mode"
            className="ml-2 rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            {dark ? "☀️" : "🌙"}
          </button>
        </nav>
      </div>
    </header>
  );
}
