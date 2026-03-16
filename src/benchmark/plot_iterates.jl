using CairoMakie
using LaTeXStrings

function plot_iterates(S::AbstractDFSolver, BP::AbstractBenchmarkProblem; invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(get_equality_manifold(BP), default_retraction_method(get_equality_manifold(BP))))
    palette = Makie.wong_colors()

    return with_theme(theme_latexfonts()) do
        res, ed, ih, oh, vs, ds = solve_problem(S, BP; invertibility_bound = invertibility_bound)
        fig = Figure(size = (500, 500))
        Axis(fig[1, 1], limits = ((-0.6, 1.8), (-2.0, 0.4)), bottomspinevisible = false, leftspinevisible = false, rightspinevisible = false, topspinevisible = false)
        hidedecorations!()

        # Contour lines
        f(x, y) = eval_obj(BP, [x, y])
        xs = LinRange(-0.6, 1.8, 100)
        ys = LinRange(-2.0, 0.4, 100)
        zs = [f(x, y) for x in xs, y in ys]
        contour!(xs, ys, zs; colormap = :thermometer, levels = -3:1:3, labels = true)

        # Circle
        arc!(Point2f(0), 1, -π, π; color = :black)

        # Parabola
        # ts = LinRange(-5, 5, 1000)
        # lines!(ts, ts .^ 2 .- 7; linewidth = 2)

        # Sum to 1
        # ts = LinRange(-5.0, 5.0, 1000)
        # lines!(ts, 1 .- ts)

        x0 = ih[1][1, :]
        scatter!([x0[1]], [x0[2]]; marker = :circle, color = palette[2], markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(x0[1] - 0.13, x0[2] - 0.1; text = L"x_0", fontsize = 20, color = palette[2])

        # x1 = ih[2][1, :]
        # scatter!([x1[1]], [x1[2]]; marker = :circle, color = palette[3], markersize = 15, strokecolor = :black, strokewidth = 1)
        # text!(x1[1] - 0.15, x1[2]; text = L"x_1", fontsize = 20, color = palette[3])

        # x2 = ih[3][1, :]
        # scatter!([x2[1]], [x2[2]]; marker = :circle, color = palette[4], markersize = 15, strokecolor = :black, strokewidth = 1)
        # text!(x2[1] - 0.04, x2[2] + 0.07; text = L"x_2", fontsize = 20, color = palette[4])

        # Iterates
        max_outer = length(ih)
        # for (it, it_data) in enumerate(ih)
        # Plot the main iterate
        it = 2
        it_data = ih[it]
        it_color = palette[mod(it, 7) + 1]
        xi = it_data[1, :]

        base_point = xi
        M = get_equality_manifold(BP)
        tdir1 = get_vector(M, base_point, [1.2])
        tdir2 = get_vector(M, base_point, [-1.2])
        xtdir1, xtdir2 = base_point .+ tdir1, base_point .+ tdir2
        lines!([xtdir1[1], xtdir2[1]], [xtdir1[2], xtdir2[2]]; color = :gray70)

        scatter!([xi[1]], [xi[2]]; marker = :circle, color = it_color, markersize = 15, strokecolor = :black, strokewidth = 1)
        # V1 scatter!([xi[1]], [xi[2]]; marker = :circle, markersize = 20, color = :transparent, strokecolor = it_color, strokewidth = 2)
        # text!(xi[1] + 0.05, xi[2] + 0.05; text = L"x_{%$(it-1)}", fontsize = 25, color = it_color)
        # V1 scatter!(it_data[2:end, 1], it_data[2:end, 2]; marker = :circle, markersize = 10, color = it_color, strokecolor = :black, strokewidth = 0.5)
        scatter!(it_data[2:end, 1], it_data[2:end, 2]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        text!(xi[1] - 0.15, xi[2]; text = L"x_1", fontsize = 20, color = it_color)
        # if it == max_outer
        #     # V1 scatter!([res[1]], [res[2]]; marker = :circle, markersize = 20, color = :transparent, strokecolor = palette[mod(it, 7) + 2], strokewidth = 2)
        #     scatter!([res[1]], [res[2]]; marker = :circle, markersize = 15, color = :black, strokecolor = :black, strokewidth = 1)
        #     # text!(res[1] + 0.05, res[2] + 0.05; text = L"x_{*}", fontsize = 25, color = palette[mod(it, 7) + 2])
        # end
        # end

        ##################### X0 ######################################
        # lines!([xtdir1[1], xtdir2[1]], [xtdir1[2], xtdir2[2]]; color = :gray70)
        # text!(xtdir2[1] - 0.25, xtdir2[2]; text = L"\mathrm{T}_{x_0}\mathbb{S}^1", fontsize = 20, color = :gray70)
        # d = ds[it]
        # scatter!(base_point .+ d; marker = :circle, markersize = 10, color = palette[2], strokewidth = 0)
        # lines!([base_point[1] + d[2, 1], it_data[2, 1]], [base_point[2] + d[2, 2], it_data[2, 2]]; linestyle = :dot, color = it_color)
        # radius = invertibility_radius(get_equality_manifold(BP), base_point)
        # rdir1 = get_vector(M, base_point, [radius])
        # rdir2 = get_vector(M, base_point, [-radius])
        # xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        # text!(xrdir1[1] - 0.06, xrdir1[2] + 0.05; text = "(", color = :gray70, fontsize = 20, rotation = -π / 2, font = :bold)
        # text!(xrdir2[1] + 0.06, xrdir2[2] - 0.05; text = "(", color = :gray70, fontsize = 20, rotation = π / 2, font = :bold)
        ################################################

        ##################### X1 ######################################
        text!(xtdir1[1] - 0.05, xtdir1[2] - 0.2; text = L"\mathrm{T}_{x_1}\mathbb{S}^1", fontsize = 20, color = :gray70)
        d = ds[it][2, :]
        bpd = base_point .+ d
        scatter!([bpd[1]], [bpd[2]]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        lines!([base_point[1] + d[1], it_data[2, 1]], [base_point[2] + d[2], it_data[2, 2]]; linestyle = :dot, color = it_color)
        radius = invertibility_radius(get_equality_manifold(BP), base_point)
        rdir1 = get_vector(M, base_point, [radius])
        rdir2 = get_vector(M, base_point, [-radius])
        xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        text!(xrdir1[1] - 0.04, xrdir1[2] + 0.04; text = "(", color = :gray70, fontsize = 20, rotation = -3π / 4, font = :bold)
        text!(xrdir2[1] + 0.04, xrdir2[2] - 0.04; text = "(", color = :gray70, fontsize = 20, rotation = π / 4, font = :bold)
        ################################################

        ##################### X2 ######################################
        # text!(xtdir1[1] - 0.45, xtdir1[2] - 0.17; text = L"\mathrm{T}_{x_2}\mathbb{S}^1", fontsize = 20, color = :gray70)
        # d = ds[it][2, :]
        # bpd = base_point .+ d
        # scatter!([bpd[1]], [bpd[2]]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # lines!([base_point[1] + d[1], it_data[2, 1]], [base_point[2] + d[2], it_data[2, 2]]; linestyle = :dot, color = it_color)
        # radius = invertibility_radius(get_equality_manifold(BP), base_point)
        # rdir1 = get_vector(M, base_point, [radius])
        # rdir2 = get_vector(M, base_point, [-radius])
        # xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        # text!(xrdir1[1], xrdir1[2] + 0.05; text = "(", color = :gray70, fontsize = 20, rotation = π, font = :bold)
        # text!(xrdir2[1], xrdir2[2] - 0.05; text = "(", color = :gray70, fontsize = 20, rotation = 0, font = :bold)
        ################################################

        ##################### X3 ######################################
        # Limits: (-1.6, 0.8), (-2.0, 0.4)
        # text!(xtdir2[1] - 0.35, xtdir2[2] - 0.05; text = L"\mathrm{T}_{x_3}\mathbb{S}^1", fontsize = 20, color = :gray70)
        # d = ds[it]
        # bpd = base_point .+ transpose(d)
        # scatter!(bpd[1, :], bpd[2, :]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # for i in 1:size(d)[1]
        #     di = d[i, :]
        #     lines!([base_point[1] + di[1], it_data[i, 1]], [base_point[2] + di[2], it_data[i, 2]]; linestyle = :dot, color = it_color)
        # end
        # # lines!([base_point[1] + d[1], it_data[2, 1]], [base_point[2] + d[2], it_data[2, 2]]; linestyle = :dot, color = it_color)
        # radius = invertibility_radius(get_equality_manifold(BP), base_point)
        # rdir1 = get_vector(M, base_point, [radius])
        # rdir2 = get_vector(M, base_point, [-radius])
        # xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        # text!(xrdir1[1] - 0.04, xrdir1[2] - 0.04; text = "(", color = :gray70, fontsize = 20, rotation = -π / 4, font = :bold)
        # text!(xrdir2[1] + 0.04, xrdir2[2] + 0.04; text = "(", color = :gray70, fontsize = 20, rotation = 3π / 4, font = :bold)

        # xstar = it_data[end, :]
        # scatter!([xstar[1]], [xstar[2]]; marker = :circle, markersize = 15, strokewidth = 1, color = :black, strokecolor = :black)
        # text!(xstar[1] + 0.03, xstar[2] + 0.03; text = L"x_{\star}", fontsize = 20, color = :black)
        ################################################

        return fig
    end
end

export plot_iterates
