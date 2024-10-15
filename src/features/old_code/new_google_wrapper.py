from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
"""Util that calls Google Scholar Search."""
from typing import Any, Dict, Optional, cast

from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class MyGoogleTrendsAPIWrapper(GoogleTrendsAPIWrapper):
    def __init__(self):
        super().__init__()

    def run(self, query: str, geo: str, date: str, tz: str, data_type: str) -> str:
        """Run query through Google Trends with Serpapi"""
        serpapi_api_key = cast(SecretStr, self.serp_api_key)
        params = {
            "engine": "google_trends",
            "api_key": serpapi_api_key.get_secret_value(),
            "q": query,
            "geo": geo,
            "date": date,
            "tz": tz,
            "data_type": data_type,
        }

        total_results = []
        client = self.serp_search_engine(params)
        client_dict = client.get_dict()
        total_results = (
            client_dict["interest_over_time"]["timeline_data"]
            if "interest_over_time" in client_dict
            else None
        )

        if not total_results:
            return "No good Trend Result was found"

        start_date = total_results[0]["date"].split()
        end_date = total_results[-1]["date"].split()
        values = [
            results.get("values")[0].get("extracted_value") for results in total_results
        ]
        min_value = min(values)
        max_value = max(values)
        avg_value = sum(values) / len(values)
        percentage_change = (
            (values[-1] - values[0])
            / (values[0] if values[0] != 0 else 1)
            * (100 if values[0] != 0 else 1)
        )

        params = {
            "engine": "google_trends",
            "api_key": serpapi_api_key.get_secret_value(),
            "data_type": "RELATED_QUERIES",
            "q": query,
            "geo": "FR",
        }

        total_results2 = {}
        client = self.serp_search_engine(params)
        total_results2 = client.get_dict().get("related_queries", {})
        rising = []
        top = []

        rising = [results.get("query") for results in total_results2.get("rising", [])]
        top = [results.get("query") for results in total_results2.get("top", [])]

        doc = [
            f"Query: {query}\n"
            f"Date From: {start_date[0]} {start_date[1]}, {start_date[-1]}\n"
            # f"Date To: {end_date[0]} {end_date[3]} {end_date[-1]}\n",
            f"Min Value: {min_value}\n"
            f"Max Value: {max_value}\n"
            f"Average Value: {avg_value}\n"
            f"Percent Change: {str(percentage_change) + '%'}\n"
            f"Trend values: {', '.join([str(x) for x in values])}\n"
            f"Rising Related Queries: {', '.join(rising)}\n"
            f"Top Related Queries: {', '.join(top)}"
        ]

        return "\n\n".join(doc)
