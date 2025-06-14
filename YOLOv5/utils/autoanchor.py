

def check_anchor_order(m):
    #* 检查anchors的顺序，如果与stride的顺序不一致，则反转
    a = m.anchors.prod(-1).mean(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da and (da.sign() != ds.sign()):
        m.anchors[:] = m.anchors.flip(0)
