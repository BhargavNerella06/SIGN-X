/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef4ff",
          100: "#d9e6ff",
          200: "#b9d0ff",
          300: "#8bb0ff",
          400: "#5c88ff",
          500: "#3763f4",
          600: "#2647d8",
          700: "#2038ae",
          800: "#1e3189",
          900: "#1c2c6b",
        },
        surface: {
          light: "#f7f8fb",
          dark: "#0b0e17",
        },
        panel: {
          light: "#ffffff",
          dark: "#131826",
        },
      },
      boxShadow: {
        card: "0 1px 2px 0 rgba(16, 24, 40, 0.06), 0 1px 3px 0 rgba(16, 24, 40, 0.10)",
      },
      keyframes: {
        pulseDot: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        fillBar: {
          from: { width: "0%" },
        },
      },
      animation: {
        "pulse-dot": "pulseDot 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
