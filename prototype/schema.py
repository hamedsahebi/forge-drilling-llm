"""Target schemas for LLM extraction.

DailyReport  - one daily drilling/completion report (any vendor template).
FigureInfo   - one figure/plot found in a document.
WellSummary  - document-level facts from long reports (e.g. End-of-Well Report).

Every field is optional: the LLM must return null rather than guess.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ActivityClass(str, Enum):
    drilling = "drilling"
    tripping = "tripping"
    casing = "casing"
    cementing = "cementing"
    circulating = "circulating"
    bop_testing = "bop_testing"
    logging_survey = "logging_survey"
    rig_up_down = "rig_up_down"
    rig_repair = "rig_repair"
    waiting = "waiting"
    wellbore_problem = "wellbore_problem"
    completion = "completion"
    other = "other"


class EventType(str, Enum):
    stuck_pipe = "stuck_pipe"
    lost_circulation = "lost_circulation"
    kick_well_control = "kick_well_control"
    tight_hole = "tight_hole"
    equipment_failure = "equipment_failure"
    bit_or_bha_failure = "bit_or_bha_failure"
    hse_incident = "hse_incident"
    weather_waiting = "weather_waiting"
    other = "other"


class ReportHeader(BaseModel):
    well_name: Optional[str] = None
    report_number: Optional[int] = Field(None, description="Report No. / RPT NUM")
    report_date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD")
    operator: Optional[str] = None
    contractor: Optional[str] = None
    rig: Optional[str] = None
    report_type: Optional[Literal["drilling", "completion"]] = None


class DepthStatus(BaseModel):
    md_ft: Optional[float] = Field(None, description="Measured depth at report time, ft")
    tvd_ft: Optional[float] = None
    footage_24h_ft: Optional[float] = None
    hole_size_in: Optional[float] = None
    current_operation: Optional[str] = None
    planned_operation: Optional[str] = None


class Casing(BaseModel):
    section: Optional[str] = Field(None, description="e.g. conductor, surface, intermediate")
    od_in: Optional[float] = None
    weight_ppf: Optional[float] = None
    grade: Optional[str] = None
    shoe_md_ft: Optional[float] = None


class MudCheck(BaseModel):
    depth_ft: Optional[float] = None
    density_ppg: Optional[float] = None
    funnel_viscosity_s: Optional[float] = None
    pv_cp: Optional[float] = None
    yp_lbf_100ft2: Optional[float] = None
    ph: Optional[float] = None
    chlorides_mg_l: Optional[float] = None


class Bit(BaseModel):
    bit_number: Optional[str] = None
    size_in: Optional[float] = None
    manufacturer: Optional[str] = None
    bit_type: Optional[str] = Field(None, description="e.g. PDC, tricone")
    serial_number: Optional[str] = None
    depth_in_ft: Optional[float] = None
    depth_out_ft: Optional[float] = None
    wob_klbf: Optional[float] = None
    rpm: Optional[float] = None
    rop_ft_hr: Optional[float] = None


class BhaComponent(BaseModel):
    order: Optional[int] = None
    component: Optional[str] = None
    od_in: Optional[float] = None
    id_in: Optional[float] = None
    length_ft: Optional[float] = None


class Survey(BaseModel):
    md_ft: Optional[float] = None
    inclination_deg: Optional[float] = None
    azimuth_deg: Optional[float] = None
    tvd_ft: Optional[float] = None


class Operation(BaseModel):
    start: Optional[str] = Field(None, description="HH:MM")
    end: Optional[str] = Field(None, description="HH:MM")
    hours: Optional[float] = None
    end_md_ft: Optional[float] = None
    activity: ActivityClass = ActivityClass.other
    is_npt: bool = Field(False, description="True only if the row is marked Non-Prod/NPT or describes unplanned "
                         "trouble: repairs, equipment failure, fishing, stuck pipe, losses, waiting on parts/weather. "
                         "Planned work (trips for bit change, rig-up, BOP tests, safety meetings, shut down for night) is NOT NPT.")
    description: Optional[str] = None


class Event(BaseModel):
    event_type: EventType
    depth_ft: Optional[float] = None
    time: Optional[str] = None
    description: str


class DailyReport(BaseModel):
    header: ReportHeader
    depth: DepthStatus
    casing: list[Casing] = []
    mud: list[MudCheck] = []
    bits: list[Bit] = []
    bha: list[BhaComponent] = []
    surveys: list[Survey] = []
    operations: list[Operation] = []
    events: list[Event] = Field([], description="Only unplanned problems or incidents (stuck pipe, losses, kicks, "
                                "failures, injuries). Routine operations are NOT events. Usually empty.")
    summary: Optional[str] = Field(None, description="One-sentence summary of the day")


# Sectioned extraction: header/depth/events are filled in one call, each table in its own call.
class PartHeader(BaseModel):
    header: ReportHeader
    depth: DepthStatus
    summary: Optional[str] = Field(None, description="One-sentence summary of the day")


class RowLabel(BaseModel):
    row: int
    is_npt: bool
    event_type: Optional[EventType] = Field(None, description="null for normal operations")


class RowLabels(BaseModel):
    labels: list[RowLabel]


class FigureKind(str, Enum):
    map = "map"
    bha_schematic = "bha_schematic"
    well_schematic = "well_schematic"
    time_vs_depth = "time_vs_depth"
    drilling_parameter_chart = "drilling_parameter_chart"
    pressure_chart = "pressure_chart"            # e.g. cement job / pressure test charts
    equipment_diagram = "equipment_diagram"      # BOPE stack, wellhead, tool drawings
    well_log = "well_log"
    mud_log = "mud_log"
    directional_plot = "directional_plot"
    photo = "photo"
    table_image = "table_image"
    other_chart = "other_chart"
    other = "other"


class Series(BaseModel):
    name: str
    unit: Optional[str] = None
    trend: Optional[str] = Field(None, description="short description of the curve's behaviour")


class FigureInfo(BaseModel):
    # transcribe -> describe -> structure: generation order follows field order, so the model first writes
    # down what it can see, then interprets it, then fills the structured fields from its own transcription
    transcription: str = Field(description="All legible labels, headings and numbers in the figure, as plain text")
    description: str = Field(description="2-3 sentence engineering description of what the figure shows")
    kind: FigureKind
    title: Optional[str] = None
    x_axis: Optional[str] = Field(None, description="label and unit")
    y_axis: Optional[str] = Field(None, description="label and unit")
    series: list[Series] = Field([], description="every curve/track/component named in the figure")
    key_values: list[str] = Field([], description="Every legible number with its meaning, each as 'name = value unit'")


class WellFacts(BaseModel):
    well_name: Optional[str] = None
    purpose: Optional[str] = None
    spud_date: Optional[str] = None
    td_date: Optional[str] = None
    total_depth_md_ft: Optional[float] = None
    total_depth_tvd_ft: Optional[float] = None
    rig: Optional[str] = None
    casing: list[Casing] = []
    bits: list[Bit] = []


class WellNarrative(BaseModel):
    events: list[Event] = []
    lessons_learned: list[str] = []


class WellSummary(BaseModel):
    well_name: Optional[str] = None
    purpose: Optional[str] = None
    spud_date: Optional[str] = None
    td_date: Optional[str] = None
    total_depth_md_ft: Optional[float] = None
    total_depth_tvd_ft: Optional[float] = None
    rig: Optional[str] = None
    casing: list[Casing] = []
    bits: list[Bit] = []
    events: list[Event] = []
    lessons_learned: list[str] = []
