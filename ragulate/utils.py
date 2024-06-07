from trulens_eval import Tru


def get_tru(recipe_name: str) -> Tru:
    Tru.RETRY_FAILED_SECONDS = 60
    Tru.RETRY_RUNNING_SECONDS = 30
    return Tru(
        database_url=f"sqlite:///{recipe_name}.sqlite", database_redact_keys=True
    )  # , name=name)
