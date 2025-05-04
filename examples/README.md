# Examples

The tea-tasting repository includes [examples](https://github.com/e10v/tea-tasting/tree/main/examples) as copies of the guides in the [marimo](https://github.com/marimo-team/marimo) notebook format. You can either download them from GitHub and run in your marimo environment, or you can run them as WASM notebooks in the online playground.

## Run in a local marimo environment

To run the examples in your marimo environment, clone the repository and change the directory:

```bash
git clone git@github.com:e10v/tea-tasting.git && cd tea-tasting
```

Install marimo, tea-tasting, and other packages used in the examples:

```bash
uv venv && uv pip install marimo tea-tasting polars ibis-framework[duckdb] tqdm
```

Launch the notebook server:

```bash
uv run marimo edit examples
```

Now you can choose and run the example notebooks.

## Run in the online playground

To run the examples as WASM notebooks in the online playground, open the following links:

- [User guide](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fuser-guide.py&embed=true).
- [Data backends](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fdata-backends.py&embed=true).
- [Power analysis](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fpower-analysis.py&embed=true).
- [Multiple hypothesis testing](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fmultiple-testing.py&embed=true).
- [Custom metrics](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fcustom-metrics.py&embed=true).
- [Simulated experiments](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fsimulated-experiments.py&embed=true).

[WASM notebooks](https://docs.marimo.io/guides/wasm/) run entirely in the browser on [Pyodide](https://github.com/pyodide/pyodide) and thus have some limitations. In particular:

- Tables and dataframes render less attractively because Pyodide doesn't always include the latest [packages versions](https://pyodide.org/en/stable/usage/packages-in-pyodide.html).
- You can't simulate experiments [in parallel](https://tea-tasting.e10v.me/simulated-experiments/#parallel-execution) because Pyodide currently [doesn't support multiprocessing](https://pyodide.org/en/stable/usage/wasm-constraints.html#included-but-not-working-modules).
- Other unpredictable issues may arise, such as the inability to use duckdb with ibis.
