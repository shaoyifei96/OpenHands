from litellm.exceptions import APIError


class CloudflareBlockError(APIError):
    """Exception raised when Cloudflare blocks the request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __instancecheck__(self, instance):
        return (
            super().__instancecheck__(instance)
            and 'Attention Required! | Cloudflare' in str(instance)
        )
