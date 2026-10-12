"""Normalization shared by dataset preparation and existing checkpoint decoders."""

# Useful variables for regression
# These were computed on the full v4 training set
LGA_MEAN = 5.233708002633516
LGA_STD = 0.2720443350888228
# These were computed on the full v9 training set
AC_MEAN = 3.067643608216848
AC_STD = 0.328575149498938
HC_MEAN = 3.1666289904036455
HC_STD = 0.27945126685955846
FL_MEAN = 1.5424865083592227
FL_STD = 0.3603675110034245
BPD_MEAN = 1.8463547431739549
BPD_STD = 0.2895954230538691


BIOMETRY_LOG_STATS = {
    "AC": (AC_MEAN, AC_STD),
    "FL": (FL_MEAN, FL_STD),
    "HC": (HC_MEAN, HC_STD),
    "BPD": (BPD_MEAN, BPD_STD),
}
