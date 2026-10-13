// ESLint 9 flat config (`npm run lint`).
import js from "@eslint/js";
import tsPlugin from "@typescript-eslint/eslint-plugin";

export default [
  { ignores: ["dist/", "node_modules/"] },
  js.configs.recommended,
  ...tsPlugin.configs["flat/recommended"],
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: "module",
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
];
