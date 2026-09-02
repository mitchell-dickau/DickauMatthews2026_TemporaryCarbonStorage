import matplotlib.patches as mpatches
import numpy as np
import statsmodels.api as sm
from matplotlib.lines import Line2D

# setup universal fig params
fig_params = {}
fig_params["colours"] = {
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274ae",
    "ssp460": "#b0724e",
    "ssp534": "#92397a",
    "ssp585": "#980002",
}
fig_params["plot_config"] = {
    "legend_fontsize": 11,
    "legend_title_fontsize": 12,
    "label_fontsize": 11,
    "subplot_title_fontsize": 12,
    "handle_length": 3.5,
    "legend_edge_color": "black",
}
fig_params["file_format"] = "pdf"
fig_params["fig_dpi"] = 500
fig_params["temp_variants"] = [
    "base",
    "sce1",
    "sce2",
    "sce3",
    "sce4",
    "sce5",
    "sce6",
]
fig_params["perm_variants"] = ["sce7", "sce8", "sce9"]


# make legend elements -- patches
def get_plot_elements(ds_tsi, variants, scenarios):
    """
    Returns the style mapping and scenario patches.
    Constructing line handles is now handled by get_line_legend_handles.
    """
    temp_pool = ["--", ":", "-.", (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (1, 1))]

    variant_styles = {}
    sorted_variants = sorted(variants)

    temp_idx = 0
    for var in sorted_variants:
        if var == "base":
            variant_styles[var] = "-"
        else:
            variant_styles[var] = temp_pool[temp_idx]
            temp_idx += 1

    # Scenario Patches (SSP1-1.9, etc.)
    patch_handles = []
    for sce in scenarios:
        formatted_label = f"SSP{sce[3]}-{sce[4]}.{sce[5]}"
        patch_handles.append(
            mpatches.Patch(color=fig_params["colours"][sce], label=formatted_label)
        )

    return variant_styles, patch_handles


# get legend elements -- lines
def get_line_legend_handles(variant_styles, plot_variants, no_titles=False):
    """
    Dynamically builds legend handles and headers based on active plot_variants.
    If no_titles is True, headers are skipped.
    """
    handles = []

    # 1. Baseline
    if "base" in plot_variants:
        handles.append(
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Baseline")
        )

    if not no_titles:
        handles.append(
            Line2D([0], [0], color="none", label=r"$\mathbf{Temporary\ Storage:}$")
        )
    for var in sorted(plot_variants):
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle=variant_styles[var],
                label=f"  Storage {var[-1]}",
            )
        )

    return handles


# plot linear regression
def plot_lr(x, y, ax, color, fill_between: bool = True, **kwargs):
    """
    Fits a linear model through the origin and calculates statistics
    using only SciPy and NumPy.
    """
    x = np.array(x)
    y = np.array(y)

    # we don't add a constant term to x because we want to force the regression through the origin
    model = sm.OLS(y, x)
    model_result = model.fit()
    # model_result.summary()
    # 4. Extract Coefficients and Confidence Intervals
    slope = model_result.params[0]
    conf_int = model_result.conf_int(alpha=0.05)  # 95% Confidence Interval

    # Calculate the centered R2 value on a zero-intercept model
    # model_result.ssr = Sum of Squared Residuals
    # model_result.centered_tss = Centered Total Sum of Squares
    r_sq = 1 - (model_result.ssr / model_result.centered_tss)

    # calculate RMSE
    rmse = np.sqrt(model_result.ssr / len(y))

    # 5. Generate Predictions and Confidence Intervals for Plotting
    # Create an array of X values covering the entire domain down to 0
    X_plot = np.linspace(0, x.max(), 100)

    # Get detailed prediction results (includes confidence intervals for the line)
    predictions = model_result.get_prediction(X_plot)
    pred_summary = predictions.summary_frame(alpha=0.05)

    # Extract fitted values and confidence limits
    y_plot_preds = pred_summary["mean"]
    ci_lower = pred_summary["obs_ci_lower"]
    ci_upper = pred_summary["obs_ci_upper"]
    """
    Obs_ci_lower and obs_ci_upper give the confidence intervals for the predicted mean response at each X value, 
    however they aren't the same as the confidence intervals for the regression line itself (which would be narrower), 
    because they aren't forced through zero. 

    Mean_ci_lower and mean_ci_upper would give the confidence intervals for the regression line, 
    and they would narrow to zero, but they are overly narrow and are not appropriate. 

    Shaded regions represent 95% prediction intervals. Because standard least-squares 
    assumes constant residual variance, these intervals do not narrow to zero at the origin.
    """

    # 6. Plotting with flexible kwargs
    kwargs.setdefault("lw", 1.5)
    kwargs.setdefault("zorder", 1)

    ax.plot(
        X_plot,
        y_plot_preds,
        color=color,
        lw=2,
        alpha=1 if fill_between else 0.5,
        label=f"Regression Line (Slope: {slope:.3f})",
    )
    if fill_between:
        # Shade the 95% Confidence Interval band
        ax.fill_between(
            X_plot,
            ci_lower,
            ci_upper,
            color=color,
            alpha=0.15,
            label="95% Confidence Interval",
        )

    return slope, conf_int, r_sq, rmse
