from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from new_google_wrapper import MyGoogleTrendsAPIWrapper as GoogleTrendsAPIWrapper

class MyGoogleTrendsQueryRun(GoogleTrendsQueryRun):
    name: str = "google_trends"
    description: str = (
        "A wrapper around Google Trends Search. "
        "Useful for when you need to get information about"
        "google search trends from Google Trends"
        "Input should be a search query."
    )
    api_wrapper: GoogleTrendsAPIWrapper

    def _run(
        self,
        query: str,
        geo: str,
        date: str,
        tz: str,
        data_type: str,

        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query, geo, date, tz, data_type)