import reflex as rx
from ui.demand_data import load_demand_rows, load_forecast_rows


class GridGuardState(rx.State):
    active_page: str = "Overview"
    replay_running: bool = False
    replay_position: int = 0

    def select_page(self, page: str):
        self.active_page = page

    def toggle_replay(self):
        self.replay_running = not self.replay_running

    def reset_replay(self):
        self.replay_position = 0
        self.replay_running = False

    def advance_replay(self):
        if self.replay_running:
            self.replay_position = min(100, self.replay_position + 1)


NAV_ITEMS = [
    ("Overview", "layout-dashboard"),
    ("Historical Replay", "radio-tower"),
    ("Demand Analysis", "chart-no-axes-combined"),
    ("Anomaly Detection", "shield-alert"),
    ("ML Performance", "brain-circuit"),
    ("Data Explorer", "database"),
]


def nav_item(label: str, icon_name: str) -> rx.Component:
    return rx.button(
        rx.hstack(
            rx.icon(tag=icon_name, size=18),
            rx.text(label, size="2"),
            width="100%",
            align="center",
            spacing="3",
        ),
        on_click=lambda: GridGuardState.select_page(label),
        variant="ghost",
        color=rx.cond(
            GridGuardState.active_page == label,
            "#EAF4FF",
            "#91A4BC",
        ),
        background=rx.cond(
            GridGuardState.active_page == label,
            "#17365C",
            "transparent",
        ),
        width="100%",
        justify="start",
        padding="12px",
        border_radius="10px",
        cursor="pointer",
    )


def metric_card(
    label: str,
    value: str,
    detail: str,
    icon_name: str,
) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.vstack(
                rx.text(label, color="#91A4BC", size="2"),
                rx.heading(value, size="7", color="#F2F7FF"),
                rx.text(detail, color="#91A4BC", size="1"),
                align="start",
                spacing="2",
            ),
            rx.spacer(),
            rx.box(
                rx.icon(tag=icon_name, size=23, color="#55B8FF"),
                background="#17365C",
                padding="12px",
                border_radius="12px",
            ),
            align="start",
            width="100%",
        ),
        background="#111F32",
        border="1px solid #24374F",
        border_radius="14px",
        padding="20px",
        width="100%",
    )


def section_card(title: str, description: str, content) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(title, size="5", color="#F2F7FF"),
            rx.text(description, color="#91A4BC", size="2"),
            content,
            align="start",
            spacing="3",
            width="100%",
        ),
        background="#111F32",
        border="1px solid #24374F",
        border_radius="14px",
        padding="20px",
        width="100%",
    )


def historical_notice() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon(tag="info", color="#F4B860", size=20),
            rx.vstack(
                rx.text(
                    "Historical dataset â€” not live monitoring",
                    color="#F4D9A7",
                    weight="bold",
                    size="2",
                ),
                rx.text(
                    "Charts and model metrics use previously recorded electricity "
                    "readings and saved evaluation results.",
                    color="#C8B994",
                    size="1",
                ),
                align="start",
                spacing="1",
            ),
            align="center",
            spacing="3",
            width="100%",
        ),
        background="#302819",
        border="1px solid #6B512A",
        border_radius="10px",
        padding="14px",
        width="100%",
    )


def quick_access_button(
    title: str,
    description: str,
    icon_name: str,
    destination: str,
    icon_color: str = "#55B8FF",
) -> rx.Component:
    return rx.button(
        rx.hstack(
            rx.box(
                rx.icon(tag=icon_name, size=23, color=icon_color),
                background="#17365C",
                padding="12px",
                border_radius="12px",
            ),
            rx.vstack(
                rx.text(title, color="#F2F7FF", weight="bold", size="2"),
                rx.text(description, color="#91A4BC", size="1"),
                align="start",
                spacing="1",
            ),
            rx.spacer(),
            rx.icon(tag="arrow-up-right", color="#91A4BC", size=18),
            align="center",
            spacing="3",
            width="100%",
        ),
        on_click=lambda: GridGuardState.select_page(destination),
        variant="outline",
        color="#F2F7FF",
        border="1px solid #304963",
        background="#13243A",
        padding="16px",
        height="auto",
        width="100%",
        justify="start",
        white_space="normal",
        _hover={
            "background": "#17365C",
            "border_color": "#55B8FF",
        },
    )



def back_to_overview_button() -> rx.Component:
    return rx.button(
        rx.icon(tag="arrow-left", size=16),
        "Back to Overview",
        on_click=lambda: GridGuardState.select_page("Overview"),
        variant="outline",
        color="#DCE8F7",
        border="1px solid #304963",
        background="#13243A",
        padding="10px 16px",
        border_radius="10px",
        _hover={"background": "#17365C", "border_color": "#55B8FF"},
    )


def sidebar() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(tag="zap", size=25, color="#FFFFFF"),
                    background="#1976D2",
                    padding="10px",
                    border_radius="12px",
                ),
                rx.vstack(
                    rx.heading("GridGuard", size="5", color="#F2F7FF"),
                    rx.text(
                        "ELECTRICITY INTELLIGENCE",
                        color="#91A4BC",
                        size="1",
                        weight="bold",
                    ),
                    align="start",
                    spacing="1",
                ),
                align="center",
                spacing="3",
                width="100%",
            ),
            rx.divider(color="#24374F"),
            rx.text("WORKSPACE", color="#657C98", size="1", weight="bold"),
            *[nav_item(label, icon_name) for label, icon_name in NAV_ITEMS],
            rx.spacer(),
            rx.box(
                rx.hstack(
                    rx.icon(tag="database", size=17, color="#55B8FF"),
                    rx.text(
                        "Historical data mode",
                        color="#C5D6E9",
                        size="2",
                    ),
                    align="center",
                    spacing="2",
                ),
                rx.text(
                    "PySpark local processing",
                    color="#91A4BC",
                    size="1",
                ),
                background="#111F32",
                border="1px solid #24374F",
                border_radius="12px",
                padding="13px",
                width="100%",
            ),
            align="start",
            spacing="4",
            height="100%",
            width="100%",
        ),
        background="#0B1422",
        border_right="1px solid #24374F",
        padding="20px",
        width="260px",
        min_width="260px",
        min_height="100vh",
    )


# OVERVIEW

def overview_page() -> rx.Component:
    forecast_rows = load_forecast_rows()

    return rx.vstack(
        rx.image(
            src="/gridguard_hero.png",
            alt="Power transmission lines and a connected electricity grid at dusk",
            width="100%",
            height="230px",
            object_fit="cover",
            border_radius="16px",
        ),
        rx.text(
            "GRIDGUARD / CONTROL ROOM",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("Electricity Intelligence", size="8", color="#F2F7FF"),
        rx.text(
            "Understand electricity consumption, compare forecasts and investigate "
            "potentially unusual usage patterns.",
            color="#91A4BC",
            size="3",
        ),

        # Keep all four requested overview metrics.
        rx.grid(
            metric_card(
                "Electricity clients",
                "370",
                "Recorded client series",
                "users",
            ),
            metric_card(
                "Historical readings",
                "51.89M",
                "Validated readings",
                "database",
            ),
            metric_card(
                "Recording interval",
                "15 min",
                "Original data frequency",
                "clock",
            ),
            metric_card(
                "Forecast model",
                "Random Forest",
                "Next-hour average demand",
                "brain-circuit",
            ),
            columns=rx.breakpoints(initial="1", sm="2", lg="4"),
            spacing="4",
            width="100%",
        ),

        section_card(
            "Actual vs Predicted Demand",
            "Random Forest next-hour demand predictions compared with actual demand "
            "and a simple previous-hour baseline.",
            rx.cond(
                len(forecast_rows) > 0,
                rx.recharts.line_chart(
                    rx.recharts.line(
                        data_key="actual",
                        name="Actual demand",
                        stroke="#2DD4BF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="predicted",
                        name="Random Forest prediction",
                        stroke="#55B8FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="baseline",
                        name="Previous-hour baseline",
                        stroke="#F4B860",
                        type="monotone",
                        stroke_dasharray="5 4",
                        dot=False,
                    ),
                    rx.recharts.x_axis(data_key="timestamp", hide=True),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(
                        stroke="#24374F",
                        stroke_dasharray="3 3",
                    ),
                    rx.recharts.tooltip(),
                    rx.recharts.legend(),
                    data=forecast_rows,
                    width="100%",
                    height=340,
                ),
                rx.text(
                    "No saved forecast test predictions were found.",
                    color="#F4B860",
                ),
            ),
        ),

        rx.text(
            "Historical electricity data (2011–2014) · Chart uses saved test predictions; "
            "it is not a live forecast.",
            color="#91A4BC",
            size="1",
        ),

        section_card(
            "Forecast model status",
            "Saved model evaluation results from the historical test period.",
            rx.vstack(
                rx.hstack(
                    rx.icon(tag="check-circle", color="#55B8FF", size=19),
                    rx.text(
                        "Random Forest evaluation results are available",
                        color="#DCE8F7",
                        size="2",
                    ),
                    align="center",
                    spacing="2",
                ),
                rx.text(
                    "MAE: 15.18 kW  •  RMSE: 22.32 kW  •  R²: 0.9902",
                    color="#DCE8F7",
                    size="2",
                ),
                align="start",
                spacing="3",
                width="100%",
            ),
        ),

        section_card(
            "Quick Access",
            "Choose a workspace below or use the persistent sidebar navigation.",
            rx.grid(
                quick_access_button(
                    "Historical Replay",
                    "Explore historical replay controls",
                    "radio-tower",
                    "Historical Replay",
                ),
                quick_access_button(
                    "Demand Analysis",
                    "View demand trends and forecast comparisons",
                    "chart-no-axes-combined",
                    "Demand Analysis",
                ),
                quick_access_button(
                    "Anomaly Detection",
                    "Inspect potential unusual readings",
                    "shield-alert",
                    "Anomaly Detection",
                    "#F4B860",
                ),
                quick_access_button(
                    "ML Performance",
                    "Review model evaluation results",
                    "brain-circuit",
                    "ML Performance",
                ),
                columns=rx.breakpoints(initial="1", sm="2"),
                spacing="3",
                width="100%",
            ),
        ),
        align="start",
        spacing="5",
        width="100%",
    )


# HISTORICAL REPLAY

def replay_page() -> rx.Component:
    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / HISTORICAL REPLAY",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("Historical Replay", size="8", color="#F2F7FF"),
        rx.text(
            "Replay controls are a UI demonstration and do not yet stream the "
            "full historical dataset.",
            color="#91A4BC",
            size="3",
        ),
        historical_notice(),
        section_card(
            "Replay controls",
            "These controls update the demonstration state only.",
            rx.vstack(
                rx.hstack(
                    rx.text("Replay position", color="#91A4BC", size="2"),
                    rx.spacer(),
                    rx.heading(
                        GridGuardState.replay_position.to_string() + "%",
                        color="#F2F7FF",
                        size="5",
                    ),
                    width="100%",
                    align="center",
                ),
                rx.progress(
                    value=GridGuardState.replay_position,
                    max=100,
                    color_scheme="blue",
                    width="100%",
                ),
                rx.hstack(
                    rx.button(
                        rx.icon(tag="play", size=16),
                        "Start / Pause",
                        on_click=GridGuardState.toggle_replay,
                        background="#1976D2",
                        color="white",
                    ),
                    rx.button(
                        rx.icon(tag="step-forward", size=16),
                        "Advance",
                        on_click=GridGuardState.advance_replay,
                        variant="outline",
                        color="#DCE8F7",
                    ),
                    rx.button(
                        rx.icon(tag="rotate-ccw", size=16),
                        "Reset",
                        on_click=GridGuardState.reset_replay,
                        variant="outline",
                        color="#DCE8F7",
                    ),
                    spacing="3",
                    wrap="wrap",
                ),
                rx.text(
                    rx.cond(
                        GridGuardState.replay_running,
                        "Replay state: running (manual demonstration)",
                        "Replay state: paused",
                    ),
                    color="#91A4BC",
                    size="2",
                ),
                align="start",
                spacing="4",
                width="100%",
            ),
        ),
        align="start",
        spacing="5",
        width="100%",
    )


# DEMAND ANALYSIS â€” CONNECTED TO SAVED CSV OUTPUTS

def demand_analysis_page() -> rx.Component:
    demand_rows = load_demand_rows()
    forecast_rows = load_forecast_rows()

    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / HISTORICAL ANALYTICS",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("Demand Analysis", size="8", color="#F2F7FF"),
        rx.text(
            "Explore recorded electricity consumption and historical forecasts.",
            color="#91A4BC",
            size="3",
        ),
        historical_notice(),
        rx.grid(
            metric_card(
                "Demand chart points",
                str(len(demand_rows)),
                "Sampled across the timeline",
                "chart-no-axes-combined",
            ),
            metric_card(
                "Forecast comparison points",
                str(len(forecast_rows)),
                "Saved test predictions",
                "brain-circuit",
            ),
            columns=rx.breakpoints(initial="1", sm="2"),
            spacing="4",
            width="100%",
        ),
        section_card(
            "Historical Average Demand",
            "Average kW across client readings. Points are sampled across the "
            "timeline and are not consecutive readings.",
            rx.cond(
                len(demand_rows) > 0,
                rx.recharts.line_chart(
                    rx.recharts.line(
                        data_key="average_kw",
                        name="Average demand (kW)",
                        stroke="#55B8FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.x_axis(data_key="timestamp", hide=True),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(
                        stroke="#24374F",
                        stroke_dasharray="3 3",
                    ),
                    rx.recharts.tooltip(),
                    data=demand_rows,
                    width="100%",
                    height=300,
                ),
                rx.text(
                    "No historical demand summary data was found.",
                    color="#F4B860",
                ),
            ),
        ),
        section_card(
            "Actual vs Predicted Demand",
            "Saved historical test results in kW: actual demand, Random Forest "
            "prediction, and previous-hour baseline.",
            rx.cond(
                len(forecast_rows) > 0,
                rx.recharts.line_chart(
                    rx.recharts.line(
                        data_key="actual",
                        name="Actual demand",
                        stroke="#F4B860",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="predicted",
                        name="Random Forest prediction",
                        stroke="#55B8FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="baseline",
                        name="Previous-hour baseline",
                        stroke="#B7A0FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.x_axis(data_key="timestamp", hide=True),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(
                        stroke="#24374F",
                        stroke_dasharray="3 3",
                    ),
                    rx.recharts.tooltip(),
                    rx.recharts.legend(),
                    data=forecast_rows,
                    width="100%",
                    height=340,
                ),
                rx.text(
                    "No forecast test predictions were found.",
                    color="#F4B860",
                ),
            ),
        ),
        rx.text(
            "Sources: saved Spark summaries and forecast test predictions. "
            "These are historical results, not live forecasts.",
            color="#91A4BC",
            size="1",
        ),
        align="start",
        spacing="5",
        width="100%",
    )


# OTHER PAGES

def placeholder_page(
    title: str,
    description: str,
    detail: str,
) -> rx.Component:
    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / WORKSPACE",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading(title, size="8", color="#F2F7FF"),
        rx.text(description, color="#91A4BC", size="3"),
        historical_notice(),
        rx.box(
            rx.icon(tag="construction", color="#F4B860", size=28),
            rx.heading(
                "Workspace ready for integration",
                size="5",
                color="#F2F7FF",
            ),
            rx.text(detail, color="#91A4BC"),
            background="#111F32",
            border="1px solid #24374F",
            border_radius="14px",
            padding="24px",
            width="100%",
        ),
        align="start",
        spacing="4",
        width="100%",
    )


def page_content() -> rx.Component:
    return rx.match(
        GridGuardState.active_page,
        ("Overview", overview_page()),
        ("Historical Replay", replay_page()),
        ("Demand Analysis", demand_analysis_page()),
        (
            "Anomaly Detection",
            placeholder_page(
                "Anomaly Detection",
                "Inspect unusual electricity readings identified by ML models.",
                "The next integration will connect the ranked anomaly investigation "
                "CSV, LOF scores, and potential-anomaly flags. Flags are not "
                "confirmed faults.",
            ),
        ),
        (
            "ML Performance",
            placeholder_page(
                "ML Performance",
                "Review model performance using saved evaluation results.",
                "Random Forest test results: MAE 15.18 kW, RMSE 22.32 kW, "
                "RÂ² 0.9902. These are historical test metrics, not live accuracy.",
            ),
        ),
        (
            "Data Explorer",
            placeholder_page(
                "Data Explorer",
                "Explore the structure and coverage of recorded electricity data.",
                "The next integration will connect processed historical data and "
                "provide a limited record preview.",
            ),
        ),
        overview_page(),
    )


def index() -> rx.Component:
    return rx.box(
        rx.hstack(
            sidebar(),
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text(
                                "GRIDGUARD INTELLIGENCE",
                                size="1",
                                color="#91A4BC",
                                weight="bold",
                            ),
                            rx.heading(
                                GridGuardState.active_page,
                                size="5",
                                color="#F2F7FF",
                            ),
                            align="start",
                            spacing="1",
                        ),
                        rx.spacer(),
                        rx.hstack(
                            rx.box(
                                width="8px",
                                height="8px",
                                border_radius="50%",
                                background="#55B8FF",
                            ),
                            rx.text(
                                "HISTORICAL MODE",
                                color="#91A4BC",
                                size="1",
                                weight="bold",
                            ),
                            align="center",
                            spacing="2",
                        ),
                        align="center",
                        width="100%",
                    ),
                    rx.divider(color="#24374F"),
                    page_content(),
                    align="start",
                    spacing="5",
                    width="100%",
                ),
                padding="28px",
                width="100%",
                min_height="100vh",
                background="#0D1828",
            ),
            align="start",
            spacing="0",
            width="100%",
        ),
        background="#0D1828",
        color="#F2F7FF",
        min_height="100vh",
        width="100%",
    )


app = rx.App()
app.add_page(index, route="/", title="GridGuard | Electricity Intelligence")