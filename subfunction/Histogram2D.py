import plotly.graph_objects as go
import numpy as np

def Histogram2D(filename, data, Image_Address, SNR=0, EVM=0, BERcount= 0):
    x = np.real(data)
    y = np.imag(data)
    miny = y.min()
    fig = go.Figure()
    filename = str(filename)
    # fig.add_trace(go.Histogram2dContour(
    #     x=x,
    #     y=y,
    #     colorscale='Hot',
    #     reversescale=True,
    #     xaxis='x',
    #     yaxis='y'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y,
    #     xaxis='x',
    #     yaxis='y',
    #     mode='markers',
    #     marker=dict(
    #         color='rgba(255,156,0,1)',
    #         size=3)
    # ))
    fig.add_trace(go.Histogram(
        y=y,
        xaxis='x2',
        marker=dict(
            color='#F58518'
        )
    ))
    fig.update_yaxes(range=[-9, 9])
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[-7, -5, -3, -1, 1, 3, 5, 7],
        )
    )


    fig.add_trace(go.Histogram(
        x=x,
        yaxis='y2',
        marker=dict(
            color='#F58518'
        )
    ))
    fig.update_xaxes(range=[-9, 9])
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[-7, -5, -3, -1, 1, 3, 5, 7],
        )
    )

    # if SNR != 0:
    #     fig.add_annotation(
    #         x=0,
    #         y=miny-0.2,
    #         text="SNR = {}(dB)".format(SNR),
    #         showarrow=False)
    #     fig.add_annotation(
    #         x=0,
    #         y=miny-0.35,
    #         text="EVM = {}(%)".format(EVM * 100),
    #         showarrow=False)
    fig.add_trace(go.Histogram2d(
        x=x,
        y=y,
        colorscale='Hot',
        nbinsx=256,
        nbinsy=256,
        zauto=True
    ))
    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False,
            fixedrange=True
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False,
            fixedrange=True
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.86, 1],
            showgrid=False,
            fixedrange=True
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.86, 1],
            showgrid=False,
            fixedrange=True
        ),
        height=800,
        width=800,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        title=go.layout.Title(text="Color Histogram---" + filename),
        xaxis_title="In-Phase",
        yaxis_title="Quadrature-Phase",
        font=dict(
            family="Time New Roman",
            size=16,
            color="Black")
        )
    if SNR !=0 :
        fig.update_layout(
            xaxis_title="SNR : {}_EVM : {}_BERcount : {}".format(SNR, EVM, BERcount)
        )
    # fig.write_image(r"data\KENG_optsim_py\20201130_DATA_2Laser_final\noLW_1GFO_noNoise_80KM_initialphase225\image\{}.png".format(filename))
    fig.write_image(Image_Address + "\{}.png".format(filename))

    # fig.show()


def Histogram2D_thesis(filename, data, Image_Address, SNR=0, EVM=0, BERcount= 0):
    x = np.real(data)
    y = np.imag(data)
    miny = y.min()
    fig = go.Figure()
    filename = str(filename)
    # fig.add_trace(go.Histogram2dContour(
    #     x=x,
    #     y=y,
    #     colorscale='Hot',
    #     reversescale=True,
    #     xaxis='x',
    #     yaxis='y'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y,
    #     xaxis='x',
    #     yaxis='y',
    #     mode='markers',
    #     marker=dict(
    #         color='rgba(255,156,0,1)',
    #         size=3)
    # ))
    fig.add_trace(go.Histogram(
        y=y,
        xaxis='x2',
        marker=dict(
            color='#F58518'
        )
    ))
    fig.update_yaxes(range=[-8.7, 8.7],showticklabels=False)
    # fig.update_layout(
    # yaxis=dict(
    #     tickmode='array'
    #     # tickvals=[-3, -1, 1, 3],
    # )
    # )

    fig.add_trace(go.Histogram(
        x=x,
        yaxis='y2',
        marker=dict(
            color='#F58518'
        ),
    ))
    fig.update_xaxes(range=[-8.7, 8.7], showticklabels=False)
    # fig.update_layout(
    # xaxis=dict(
    #     tickmode='array'
    #     # tickvals=[-3, -1, 1, 3],
    # )
    showticklabels=False
    # )

    # if SNR != 0:
    #     print(miny)
    #     print(x.max())
    #     fig.add_annotation(
    #         x=x.max() + 0.5,
    #         y=y.max() + 0.5,
    #         text="EVM = {}(%)".format(EVM * 100),
    #         showarrow=False)
    #     fig.add_annotation(
    #         x=0,
    #         y=miny-0.35,
    #         text="SNR = {}(dB)".format(SNR),
    #         showarrow=False)
    fig.add_trace(go.Histogram2d(
        x=x,
        y=y,
        colorscale='Hot',
        nbinsx=256,
        nbinsy=256,
        zauto=True
    ))
    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False,
            fixedrange=True
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False,
            fixedrange=True
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.86, 1],
            showgrid=False,
            fixedrange=True
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.86, 1],
            showgrid=False,
            fixedrange=True
        ),
        height=800,
        width=800,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        title=go.layout.Title(text="Color Histogram---" + filename),
        xaxis_title="In-Phase",
        yaxis_title="Quadrature-Phase",
        font=dict(
            family="Time New Roman",
            size=25,
            color="Black")
    )
    # if SNR !=0 :
    #     fig.update_layout(
    #         xaxis_title="SNR : {}_EVM : {}_BERcount : {}".format(SNR, EVM, BERcount)
    #     )

        # fig.write_image(r"data\KENG_optsim_py\20201130_DATA_2Laser_final\noLW_1GFO_noNoise_80KM_initialphase225\image\{}.png".format(filename))
    fig.write_image(Image_Address + "\{}.png".format(filename))