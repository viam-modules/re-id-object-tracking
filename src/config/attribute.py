from typing import Any, List, Optional

from viam.proto.app.robot import ServiceConfig


class Attribute:
    def __init__(
        self,
        field_name: str,
        required: bool = False,
        default_value: Optional[Any] = None,
    ):
        self.field_name = field_name
        self.required = required
        self.default_value = default_value

    def validate(self, config: "ServiceConfig", value=None):
        if value is not None:
            return value
        value = config.attributes.fields.get(self.field_name, self.default_value)
        if self.required and value is None:
            raise ValueError(
                f"Missing required configuration attribute: {self.field_name}"
            )
        return value


class IntAttribute(Attribute):
    def __init__(
        self,
        field_name: str,
        required: bool = False,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        default_value: Optional[int] = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(field_name, required, default_value)

    def validate(self, config: "ServiceConfig", value=None):
        value = super().validate(config, value)
        if not isinstance(value, (float, int)):
            if not hasattr(value, "number_value"):
                raise ValueError(
                    f"Expected number for '{self.field_name}', got {type(value).__name__}"
                )
            value = value.number_value
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                f"Expected integer for '{self.field_name}', but got float with a decimal part."
            )
        if self.min_value is not None and value < self.min_value:
            raise ValueError(
                f"Value for '{self.field_name}' should be at least {self.min_value}. Got {value}."
            )
        if self.max_value is not None and value > self.max_value:
            raise ValueError(
                f"Value for '{self.field_name}' should be at most {self.max_value}. Got {value}."
            )
        return int(value)


class FloatAttribute(Attribute):
    def __init__(
        self,
        field_name: str,
        required: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        default_value: Optional[float] = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(field_name, required, default_value)

    def validate(self, config: "ServiceConfig", value=None):
        value = super().validate(config, value)
        if not isinstance(value, (float, int)):
            if not hasattr(value, "number_value"):
                raise ValueError(
                    f"Expected number for '{self.field_name}', got {type(value).__name__}"
                )
            value = float(value.number_value)

        if self.min_value is not None and value < self.min_value:
            raise ValueError(
                f"Value for '{self.field_name}' should be at least {self.min_value}. Got {value}."
            )
        if self.max_value is not None and value > self.max_value:
            raise ValueError(
                f"Value for '{self.field_name}' should be at most {self.max_value}. Got {value}."
            )
        return value


class StringAttribute(Attribute):
    def __init__(
        self,
        field_name: str,
        required: bool = False,
        allowlist: Optional[list] = None,
        default_value: Optional[str] = None,
    ):
        self.allowlist = allowlist
        super().__init__(field_name, required, default_value)

    def validate(self, config: "ServiceConfig", value=None):
        value = super().validate(config, value)
        if value is None:
            return value
        if not isinstance(value, str):  # if it's not the default value
            if not hasattr(
                value, "string_value"
            ):  # if it's from the config but the wrong kind
                raise ValueError(
                    f"Expected string for '{self.field_name}', got {type(value).__name__}"
                )

            value = str(value.string_value)

        if self.allowlist and value not in self.allowlist:
            raise ValueError(
                f"Invalid value '{value}' for '{self.field_name}'. Allowed values are: {self.allowlist}."
            )
        return value

    def __str__(self):
        return self.value


class BoolAttribute(Attribute):
    def __init__(
        self,
        field_name: str,
        required: bool = False,
        default_value: Optional[bool] = None,
    ):
        super().__init__(field_name, required, default_value)

    def validate(self, config: "ServiceConfig", value=None):
        value = super().validate(config, value)
        if not isinstance(value, bool):  # if it's not the default value
            if not hasattr(
                value, "bool_value"
            ):  # if it's from the config but the wrong kind
                raise ValueError(
                    f"Expected string for '{self.field_name}', got {type(value).__name__}"
                )
            value = value.bool_value
        return value

    def __bool__(self):
        return self.value


class DictAttribute(Attribute):
    def __init__(
        self,
        field_name: str,
        required: bool = False,
        fields: Optional[List["Attribute"]] = None,
        default_value: Optional[dict] = None,
    ):
        self.fields = fields or []
        super().__init__(field_name, required, default_value)

    def validate(self, config: "ServiceConfig", value=None):
        value = super().validate(config)
        if value is None:
            return value
        value = dict(value.struct_value.fields)
        for attribute in self.fields:
            value[attribute.field_name] = attribute.validate(
                config, value[attribute.field_name]
            )

        return value
