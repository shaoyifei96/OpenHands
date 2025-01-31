from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from openhands.core.logger import openhands_logger as logger
from openhands.utils.tenacity_stop import stop_if_should_exit


class RetryMixin:
    """Mixin class for retry logic."""

    def retry_decorator(self, **kwargs):
        """
        Create a LLM retry decorator with customizable parameters. This is used for 429 errors, and a few other exceptions in LLM classes.

        Args:
            **kwargs: Keyword arguments to override default retry behavior.
                      Keys: num_retries, retry_exceptions, exclude_exceptions, retry_min_wait, retry_max_wait, retry_multiplier

        Returns:
            A retry decorator with the parameters customizable in configuration.
        """
        num_retries = kwargs.get('num_retries')
        retry_exceptions: tuple = kwargs.get('retry_exceptions', ())
        exclude_exceptions: tuple = kwargs.get('exclude_exceptions', ())
        retry_min_wait = kwargs.get('retry_min_wait')
        retry_max_wait = kwargs.get('retry_max_wait')
        retry_multiplier = kwargs.get('retry_multiplier')

        def _filter_exceptions(e):
            # First check if the exception is in the exclude list
            if isinstance(e, exclude_exceptions):
                return False

            # Then check if it's in the retry list
            return isinstance(e, retry_exceptions)

        return retry(
            before_sleep=self.log_retry_attempt,
            stop=stop_after_attempt(num_retries) | stop_if_should_exit(),
            reraise=True,
            # Use the above filter function to decide whether to retry or not.
            retry=retry_if_exception(_filter_exceptions),
            wait=wait_exponential(
                multiplier=retry_multiplier,
                min=retry_min_wait,
                max=retry_max_wait,
            ),
        )

    def log_retry_attempt(self, retry_state):
        """Log retry attempts."""
        exception = retry_state.outcome.exception()
        logger.error(
            f'{exception}. Attempt #{retry_state.attempt_number} | You can customize retry values in the configuration.',
            exc_info=False,
        )
