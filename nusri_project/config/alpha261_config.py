from __future__ import annotations

import math
from typing import Dict, List, Tuple

from qlib.contrib.data.loader import Alpha158DL


def _zscore(expr: str, n: int) -> str:
    return f"({expr}-Mean({expr},{n}))/(Std({expr},{n})+1e-12)"


def _tr_expr() -> str:
    return (
        "Greater($high-$low,Greater(Abs($high-Ref($close,1)),Abs($low-Ref($close,1))))"
    )


def _alpha158_expr_map() -> Dict[str, str]:
    # Mirror qlib.contrib.data.handler.Alpha158.get_feature_config()
    # NOTE: we do NOT require a raw 'vwap' column; any $vwap reference is rewritten to amount/volume.
    conf = {
        "kbar": {},
        "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]},
        "rolling": {},
    }
    fields, names = Alpha158DL.get_feature_config(conf)
    expr_map = dict(zip(names, fields))

    vwap_expr = "($amount/($volume+1e-12))"
    for k, v in list(expr_map.items()):
        if "$vwap" in v:
            expr_map[k] = v.replace("$vwap", vwap_expr)
    return expr_map


def _new_factor_exprs() -> List[Tuple[str, str]]:
    ret1 = "Log($close/Ref($close,1))"
    tr = _tr_expr()
    range_log = "Log($high/$low)"
    body = "Abs($close-$open)/($high-$low+1e-12)"
    up_shadow = "($high-Greater($open,$close))/($high-$low+1e-12)"
    low_shadow = "(Less($open,$close)-$low)/($high-$low+1e-12)"
    vwap20 = "Sum($amount,20)/(Sum($volume,20)+1e-12)"
    vwap60 = "Sum($amount,60)/(Sum($volume,60)+1e-12)"

    def ema_spread(a: int, b: int) -> str:
        return f"EMA($close,{a})-EMA($close,{b})"

    def momz(k: int) -> str:
        return f"Log($close/Ref($close,{k}))/Std({ret1},{k})"

    def d2h(n: int) -> str:
        return f"$close/Max($high,{n})-1"

    def d2l(n: int) -> str:
        return f"$close/Min($low,{n})-1"

    def donch(n: int) -> str:
        return f"($close-Max($high,{n}))/(Std($close,{n})+1e-12)"

    def atr(n: int) -> str:
        return f"Mean({tr},{n})"

    def natr(n: int) -> str:
        return f"{atr(n)}/$close"

    def rv(n: int) -> str:
        return f"Std({ret1},{n})"

    def vov(n: int) -> str:
        return f"Std({rv(20)},{n})"

    def pkvol(n: int) -> str:
        return f"Mean(Log($high/$low)*Log($high/$low),{n})"

    def gkvol(n: int) -> str:
        gk_const = 2 * math.log(2) - 1
        return (
            "Mean("
            "0.5*(Log($high/$low)*Log($high/$low))"
            f"-{gk_const:.15f}*(Log($close/$open)*Log($close/$open))"
            f",{n})"
        )

    def body_n(n: int) -> str:
        return body if n == 1 else f"Mean({body},{n})"

    def up_shadow_n(n: int) -> str:
        return up_shadow if n == 1 else f"Mean({up_shadow},{n})"

    def low_shadow_n(n: int) -> str:
        return low_shadow if n == 1 else f"Mean({low_shadow},{n})"

    def rangez(n: int) -> str:
        return _zscore(range_log, n)

    def volz(n: int) -> str:
        return _zscore("$volume", n)

    def amtz(n: int) -> str:
        return _zscore("$amount", n)

    def vwapn(n: int) -> str:
        return vwap20 if n == 20 else vwap60

    def vwapdev(n: int) -> str:
        return f"Log($close/{vwapn(n)})"

    def obv_z(n: int) -> str:
        obv = f"Sum(Sign($close-Ref($close,1))*$volume,{n})"
        return _zscore(obv, n)

    def pvt_z(n: int) -> str:
        pvt = f"Sum(($close/Ref($close,1)-1)*$volume,{n})"
        return _zscore(pvt, n)

    def illiq(n: int) -> str:
        return f"Mean(Abs({ret1})/($amount+1e-12),{n})"

    def vwmom(n: int) -> str:
        return f"Sum({ret1}*$amount,{n})/(Sum($amount,{n})+1e-12)"

    def er(n: int) -> str:
        return f"Abs($close-Ref($close,{n}))/(Sum(Abs($close-Ref($close,1)),{n})+1e-12)"

    def chop(n: int) -> str:
        log_n = math.log(n)
        return (
            f"100*Log(Sum({tr},{n})/(Max($high,{n})-Min($low,{n})+1e-12))/{log_n:.15f}"
        )

    def adx(n: int) -> str:
        up_move = "($high-Ref($high,1))"
        down_move = "(Ref($low,1)-$low)"
        plus_dm = f"If(And(Gt({up_move},{down_move}),Gt({up_move},0)),{up_move},0)"
        minus_dm = f"If(And(Gt({down_move},{up_move}),Gt({down_move},0)),{down_move},0)"
        atr_n = f"Sum({tr},{n})"
        plus_di = f"100*Sum({plus_dm},{n})/({atr_n}+1e-12)"
        minus_di = f"100*Sum({minus_dm},{n})/({atr_n}+1e-12)"
        dx = f"100*Abs({plus_di}-{minus_di})/({plus_di}+{minus_di}+1e-12)"
        return f"Mean({dx},{n})"

    def aroon(n: int) -> str:
        aroon_up = f"100*({n}-IdxMax($high,{n}))/{n}"
        aroon_down = f"100*({n}-IdxMin($low,{n}))/{n}"
        return f"{aroon_up}-{aroon_down}"

    def mdd(n: int) -> str:
        dd = f"1-$close/Max($close,{n})"
        return f"Max({dd},{n})"

    def ddcur(n: int) -> str:
        return f"1-$close/Max($close,{n})"

    def skew(n: int) -> str:
        return f"Skew({ret1},{n})"

    def kurt(n: int) -> str:
        return f"Kurt({ret1},{n})"

    def ind_pos(expr: str) -> str:
        return f"(Abs({expr})+({expr}))/(2*Abs({expr})+1e-12)"

    def ind_neg(expr: str) -> str:
        return f"(Abs({expr})-({expr}))/(2*Abs({expr})+1e-12)"

    buy_base = "$taker_buy_base_volume"
    buy_quote = "$taker_buy_quote_volume"
    sell_base = f"($volume-{buy_base})"
    sell_quote = f"($amount-{buy_quote})"
    vwap_bar = "$amount/($volume+1e-12)"
    ti = f"(({buy_base})-({sell_base}))/($volume+1e-12)"
    di = f"(({buy_quote})-({sell_quote}))/($amount+1e-12)"
    stf = f"(({buy_quote})-({sell_quote}))"

    # Funding updates every 8 hours; prefer window sizes divisible by 8.
    fz72 = _zscore("$funding_rate", 72)  # 3 days
    fz240 = _zscore("$funding_rate", 240)  # 10 days
    fund_pos = ind_pos("$funding_rate")

    exprs = [
        # --- Existing (previous alpha170) price/volume factors ---
        ("RET1", ret1),
        ("RET5", "Log($close/Ref($close,5))"),
        ("RET10", "Log($close/Ref($close,10))"),
        ("RET20", "Log($close/Ref($close,20))"),
        ("RET60", "Log($close/Ref($close,60))"),
        ("EMASPREAD12_26", ema_spread(12, 26)),
        ("EMASPREAD6_24", ema_spread(6, 24)),
        ("MOMZ20", momz(20)),
        ("MOMZ60", momz(60)),
        ("D2H20", d2h(20)),
        ("D2H60", d2h(60)),
        ("D2L20", d2l(20)),
        ("D2L60", d2l(60)),
        ("DONCHBREAK20", donch(20)),
        ("DONCHBREAK60", donch(60)),
        ("ATR14", atr(14)),
        ("ATR30", atr(30)),
        ("ATR60", atr(60)),
        ("NATR14", natr(14)),
        ("NATR30", natr(30)),
        ("RV20", rv(20)),
        ("RV60", rv(60)),
        ("VOV20", vov(20)),
        ("VOV60", vov(60)),
        ("PKVOL20", pkvol(20)),
        ("PKVOL60", pkvol(60)),
        ("GKVOL20", gkvol(20)),
        ("GKVOL60", gkvol(60)),
        ("BODY1", body_n(1)),
        ("BODY5", body_n(5)),
        ("BODY20", body_n(20)),
        ("UPSHADOW1", up_shadow_n(1)),
        ("UPSHADOW5", up_shadow_n(5)),
        ("UPSHADOW20", up_shadow_n(20)),
        ("LOWSHADOW1", low_shadow_n(1)),
        ("LOWSHADOW5", low_shadow_n(5)),
        ("LOWSHADOW20", low_shadow_n(20)),
        ("RANGEZ20", rangez(20)),
        ("RANGEZ60", rangez(60)),
        ("VOLZ20", volz(20)),
        ("VOLZ60", volz(60)),
        ("AMTZ20", amtz(20)),
        ("AMTZ60", amtz(60)),
        ("VWAPN20", vwapn(20)),
        ("VWAPN60", vwapn(60)),
        ("VWAPDEV20", vwapdev(20)),
        ("VWAPDEV60", vwapdev(60)),
        ("OBV20", obv_z(20)),
        ("OBV60", obv_z(60)),
        ("PVT20", pvt_z(20)),
        ("PVT60", pvt_z(60)),
        ("ILLIQ20", illiq(20)),
        ("ILLIQ60", illiq(60)),
        ("VWMOM20", vwmom(20)),
        ("VWMOM60", vwmom(60)),
        ("ER10", er(10)),
        ("ER20", er(20)),
        ("ER60", er(60)),
        ("CHOP14", chop(14)),
        ("CHOP30", chop(30)),
        ("ADX14", adx(14)),
        ("ADX30", adx(30)),
        ("AROON20", aroon(20)),
        ("AROON60", aroon(60)),
        ("MDD20", mdd(20)),
        ("MDD60", mdd(60)),
        ("DDCUR20", ddcur(20)),
        ("DDCUR60", ddcur(60)),
        ("SKEW60", skew(60)),
        ("KURT60", kurt(60)),
        # --- Order-flow (taker) derived factors ---
        ("BUY_BASE", buy_base),
        ("SELL_BASE", sell_base),
        ("BUY_QUOTE", buy_quote),
        ("SELL_QUOTE", sell_quote),
        ("VWAP_BAR", vwap_bar),
        ("TBR", f"({buy_base})/($volume+1e-12)"),
        ("TI", ti),
        ("DI", di),
        ("STF", stf),
        ("STFZ72", _zscore(stf, 72)),
        ("FMO24", f"Mean({ti},24)"),
        ("FMO72", f"Mean({ti},72)"),
        ("FA8_24", f"Mean({ti},8)-Mean({ti},24)"),
        ("FA24_72", f"Mean({ti},24)-Mean({ti},72)"),
        ("FRD72", f"({ti})-({_zscore(ret1, 72)})"),
        # This factor captures the divergence between trade flow and price movement. If the Taker Intensity (TI) is high while the Z-score of Returns remains low, it indicates the presence of a Hidden Liquidity Wall or significant passive selling pressure. This serves as a powerful signal for potential market tops."
        ("IMP", f"Abs({ret1})/($amount+1e-12)"),
        # This is a measure of Market Impact, representing the price change per unit of trading volume. A higher value indicates thinner liquidity, meaning the market is more fragile. In BTC timing, a sudden spike in IMP often foreshadows an expansion in volatility or a potential liquidity cascade."
        ("ABNR", f"({ti})*({ind_neg(ret1)})"),
        ("ASPR", f"(0-({ti}))*({ind_pos(ret1)})"),
        # --- Funding rate factors ---
        ("FZ72", fz72),
        ("FZ240", fz240),
        ("FCHG8", "$funding_rate-Ref($funding_rate,8)"),
        ("FCHG24", "$funding_rate-Ref($funding_rate,24)"),
        ("FREG", fund_pos),
        ("FCNT72", f"Sum({fund_pos},72)"),
        ("FCNT240", f"Sum({fund_pos},240)"),
        ("FFA72", f"({fz72})*({ti})"),
        ("FRG5_72", f"({fz72})-({_zscore('Log($close/Ref($close,5))', 72)})"),
        ("FRG20_72", f"({fz72})-({_zscore('Log($close/Ref($close,20))', 72)})"),
        # --- Composite / tradable structure factors ---
        ("BRK_FLOW20_8", f"{ind_pos('$close-Ref(Max($high,20),1)')}*Mean({ti},8)"),
        ("REV_CL", f"{ind_pos(f'({fz72})-1.5')}*{ind_neg(ti)}*({ret1})"),
        ("SQZ", f"{ind_pos(f'(0-({fz72})-1.5)')}*{ind_pos(f'Mean({ti},24)')}"),
        ("LTT", f"{_zscore(f'Abs({ret1})/($amount+1e-12)', 72)}*Sign(Mean({ret1},24))"),
    ]
    return [(expr, name) for name, expr in exprs]


def get_alpha261_config() -> Tuple[List[str], List[str]]:
    expr_map = _alpha158_expr_map()
    new_factors = _new_factor_exprs()

    exprs: List[str] = []
    names: List[str] = []

    for name, expr in expr_map.items():
        names.append(name)
        exprs.append(expr)

    for expr, name in new_factors:
        if name in names:
            raise ValueError(f"duplicate factor name: {name}")
        names.append(name)
        exprs.append(expr)

    return exprs, names


def get_top23_config() -> Tuple[List[str], List[str]]:
    expr_map = _alpha158_expr_map()
    new_factors = {name: expr for expr, name in _new_factor_exprs()}

    top23_names = [
        "ADX14",
        "ADX30",
        "CHOP14",
        "CHOP30",
        "CORR20",
        "FMO72",
        "FRG20_72",
        "FZ240",
        "FZ72",
        "GKVOL60",
        "IMAX30",
        "IMXD30",
        "LOWSHADOW20",
        "MDD60",
        "NATR30",
        "OBV20",
        "PVT60",
        "RV20",
        "RV60",
        "STD60",
        "VOV60",
        "VWAPDEV60",
        "WVMA60",
    ]

    exprs: List[str] = []
    names: List[str] = []

    for name in top23_names:
        if name in expr_map:
            expr = expr_map[name]
        elif name in new_factors:
            expr = new_factors[name]
        else:
            raise ValueError(f"unknown factor name: {name}")
        names.append(name)
        exprs.append(expr)

    return exprs, names


def get_alpha261_feature_count() -> int:
    exprs, _ = get_alpha261_config()
    return len(exprs)


def get_alpha_feature_config() -> Tuple[List[str], List[str]]:
    return get_alpha261_config()
