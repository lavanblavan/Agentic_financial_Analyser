"""Typed contracts between agents. Do not pass raw strings across the A→B boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RiskFactor(BaseModel):
    name: str
    severity: Literal["low", "medium", "high"]
    horizon: str = "90d"
    evidence: str


class DataBrief(BaseModel):
    ticker: str
    company_name: str = ""
    current_price: float
    vol_30d_pct: float
    vol_90d_pct: float | None = None
    momentum: str
    rsi_14: float | None = None
    financial_health: str = ""
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: str = "neutral"
    headlines: list[str] = Field(default_factory=list)
    quantitative_risks: list[RiskFactor] = Field(default_factory=list)
    hedge_strategy: str = ""
    notes: str = ""


RequestKind = Literal["vol_90", "web_search", "news", "sentiment", "revise"]


class CritiqueDecision(BaseModel):
    need_more_data: bool
    request: str | None = None
    request_kind: RequestKind | None = None
    reason: str
    issues: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    ticker: str
    financial_health: str
    market_sentiment: str
    top_risks: list[RiskFactor]
    hedge_or_strategy: str
    used_90d_vol: bool = False
    critique_rounds: int = 0
    critic_notes: str = ""
    brief: DataBrief
