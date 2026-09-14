using CairoMakie
using LaTeXStrings
using PRIMA

function plot_iterates(S::AbstractDFSolver, BP::AbstractBenchmarkProblem; invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(get_equality_manifold(BP), default_retraction_method(get_equality_manifold(BP))))
    palette = Makie.wong_colors()

    return with_theme(theme_latexfonts()) do
        res, ed, ih, oh, vs, ds, xs = solve_problem(S, BP; invertibility_bound = invertibility_bound)
        fig = Figure(size = (500, 500))
        Axis(fig[1, 1], limits = ((-2.0, 2.0), (-2.0, 2.0)), bottomspinevisible = false, leftspinevisible = false, rightspinevisible = false, topspinevisible = false, aspect = DataAspect())
        hidedecorations!()

        # Contour lines
        f(x, y) = eval_obj(BP, [x, y])
        xs = LinRange(-2.0, 2.0, 200)
        ys = LinRange(-2.0, 2.0, 200)
        zs = [f(x, y) for x in xs, y in ys]
        contour!(xs, ys, zs; colormap = :thermometer, levels = -5:1:5, labels = true)

        # Circle
        arc!(Point2f(0), 1, -π, π; color = :black)

        # Parabola
        # ts = LinRange(-5, 5, 1000)
        # lines!(ts, ts .^ 2 .- 7; linewidth = 2)

        # Sum to 1
        # ts = LinRange(-5.0, 5.0, 1000)
        # lines!(ts, 1 .- ts)

        # Iterates
        # max_outer = length(ih)
        # for (it, it_data) in enumerate(ih)
        # Plot the main iterate
        it = 4
        it_data = ih[it]
        it_color = palette[mod(it, 7) + 1]
        xi = it_data[1, :]

        base_point = xi
        M = get_equality_manifold(BP)
        tdir1 = get_vector(M, base_point, [-1.5])
        tdir2 = get_vector(M, base_point, [1.5])
        xtdir1, xtdir2 = base_point .+ tdir1, base_point .+ tdir2
        lines!([xtdir1[1], xtdir2[1]], [xtdir1[2], xtdir2[2]]; color = :gray70)

        x0 = ih[1][1, :]
        scatter!([x0[1]], [x0[2]]; marker = :circle, color = palette[2], markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(x0[1] + 0.1, x0[2] - 0.05; text = L"x_0", fontsize = 25, color = palette[2])

        x1 = ih[2][1, :]
        scatter!([x1[1]], [x1[2]]; marker = :circle, color = palette[3], markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(x1[1] + 0.1, x1[2] - 0.2; text = L"x_1", fontsize = 25, color = palette[3])

        x2 = ih[3][1, :]
        scatter!([x2[1]], [x2[2]]; marker = :circle, color = palette[4], markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(x2[1] - 0.08, x2[2] - 0.27; text = L"x_2", fontsize = 25, color = palette[4])

        x3 = ih[4][1, :]
        scatter!([x3[1]], [x3[2]]; marker = :circle, color = palette[5], markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(x3[1] - 0.15, x3[2] - 0.27; text = L"x_3", fontsize = 25, color = palette[5])

        # V1 scatter!([xi[1]], [xi[2]]; marker = :circle, markersize = 20, color = :transparent, strokecolor = it_color, strokewidth = 2)
        # text!(xi[1] + 0.05, xi[2] + 0.05; text = L"x_{%$(it-1)}", fontsize = 25, color = it_color)
        # V1 scatter!(it_data[2:end, 1], it_data[2:end, 2]; marker = :circle, markersize = 10, color = it_color, strokecolor = :black, strokewidth = 0.5)
        # scatter!(it_data[2:end, 1], it_data[2:end, 2]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # text!(xi[1] + 0.1, xi[2] + 0.03; text = L"x_{%$(it-1)}", fontsize = 25, color = it_color)
        # if it == max_outer
        #     # V1 scatter!([res[1]], [res[2]]; marker = :circle, markersize = 20, color = :transparent, strokecolor = palette[mod(it, 7) + 2], strokewidth = 2)
        #     scatter!([res[1]], [res[2]]; marker = :circle, markersize = 15, color = :black, strokecolor = :black, strokewidth = 1)
        #     text!(res[1] + 0.05, res[2] + 0.05; text = L"x_{\star}", fontsize = 25, color = :black)
        # end

        ##################### X0 ######################################
        # lines!([xtdir1[1], xtdir2[1]], [xtdir1[2], xtdir2[2]]; color = :gray70)
        # scatter!([xi[1]], [xi[2]]; marker = :circle, color = it_color, markersize = 15, strokecolor = :black, strokewidth = 1)
        # text!(xtdir1[1] - 0.5, xtdir1[2] + 0.2; text = L"\mathrm{T}_{x_0}\mathbb{S}^1", fontsize = 25, color = :gray70)

        # d = ds[it][2:end, :]
        # radius = invertibility_radius(M, base_point, default_retraction_method(M), invertibility_bound)
        # rdir1 = get_vector(M, base_point, [radius])
        # rdir2 = get_vector(M, base_point, [-radius])
        # xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # # # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        # text!(xrdir1[1] - 0.09, xrdir1[2]; text = "(", color = :gray70, fontsize = 20, rotation = -π / 2, font = :bold)
        # text!(xrdir2[1] + 0.09, xrdir2[2]; text = "(", color = :gray70, fontsize = 20, rotation = π / 2, font = :bold)

        # bpd = base_point .+ transpose(d)
        # scatter!(it_data[:, 1], it_data[:, 2]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # scatter!(bpd[1, :], bpd[2, :]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # for i in 1:size(bpd)[2]
        #     di = bpd[:, i]
        #     lines!([di[1], it_data[i + 1, 1]], [di[2], it_data[i + 1, 2]]; linestyle = :dot, color = it_color)
        # end
        ################################################

        ################################################
        # scatter!([xi[1]], [xi[2]]; marker = :circle, color = it_color, markersize = 15, strokecolor = :black, strokewidth = 1)
        # text!(xi[1] - 0.2, xi[2] - 0.27; text = L"x_{%$(it-1)}", fontsize = 25, color = it_color)
        # scatter!(it_data[:, 1], it_data[:, 2]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # ################################################

        # scatter!([res[1]], [res[2]]; marker = :circle, markersize = 15, strokewidth = 1, color = :black, strokecolor = :black)
        # text!(res[1] - 0.25, res[2] - 0.25; text = L"x_{\star}", fontsize = 30, color = :black)

        ##################### X1 ######################################
        # Limits: (-0.6, 1.8) and (-2.0, 0.4)
        # text!(xtdir1[1] + 0.12, xtdir1[2] - 0.15; text = L"\mathrm{T}_{x_1}\mathbb{S}^1", fontsize = 25, color = :gray70)
        # d = ds[it][2, :]
        # bpd = base_point .+ d
        # scatter!([bpd[1]], [bpd[2]]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # lines!([base_point[1] + d[1], it_data[2, 1]], [base_point[2] + d[2], it_data[2, 2]]; linestyle = :dot, color = it_color)
        # radius = invertibility_radius(get_equality_manifold(BP), base_point)
        # rdir1 = get_vector(M, base_point, [radius])
        # rdir2 = get_vector(M, base_point, [-radius])
        # xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # # # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        # text!(xrdir1[1] - 0.07, xrdir1[2] + 0.07; text = "(", color = :gray70, fontsize = 20, rotation = -3π / 4, font = :bold)
        # text!(xrdir2[1] + 0.07, xrdir2[2] - 0.07; text = "(", color = :gray70, fontsize = 20, rotation = π / 4, font = :bold)
        ################################################

        ##################### X2 ######################################
        # text!(xtdir1[1] - 0.45, xtdir1[2] - 0.05; text = L"\mathrm{T}_{x_2}\mathbb{S}^1", fontsize = 25, color = :gray70)
        # d = ds[it][2, :]
        # bpd = base_point .+ d
        # scatter!([bpd[1]], [bpd[2]]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        # lines!([base_point[1] + d[1], it_data[2, 1]], [base_point[2] + d[2], it_data[2, 2]]; linestyle = :dot, color = it_color)
        # radius = invertibility_radius(get_equality_manifold(BP), base_point)
        # rdir1 = get_vector(M, base_point, [radius])
        # rdir2 = get_vector(M, base_point, [-radius])
        # xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # # # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        # text!(xrdir1[1], xrdir1[2] + 0.1; text = "(", color = :gray70, fontsize = 20, rotation = π, font = :bold)
        # text!(xrdir2[1], xrdir2[2] - 0.11; text = "(", color = :gray70, fontsize = 20, rotation = 0, font = :bold)
        ################################################

        ##################### X3 ######################################
        # Limits: (-1.6, 0.8), (-2.0, 0.4)
        text!(xtdir2[1] + 0.05, xtdir2[2]; text = L"\mathrm{T}_{x_3}\mathbb{S}^1", fontsize = 25, color = :gray70)
        d = ds[it]
        bpd = base_point .+ transpose(d)
        k = size(d)[1]
        scatter!(bpd[1, 1:k], bpd[2, 1:k]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        scatter!(it_data[1:k, 1], it_data[1:k, 2]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        for i in 1:k
            di = d[i, :]
            lines!([base_point[1] + di[1], it_data[i, 1]], [base_point[2] + di[2], it_data[i, 2]]; linestyle = :dot, color = it_color)
        end
        # # lines!([base_point[1] + d[1], it_data[2, 1]], [base_point[2] + d[2], it_data[2, 2]]; linestyle = :dot, color = it_color)
        radius = invertibility_radius(get_equality_manifold(BP), base_point)
        rdir1 = get_vector(M, base_point, [radius])
        rdir2 = get_vector(M, base_point, [-radius])
        xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        # This works: scatter!([xrdir1[1], xrdir2[1]], [xrdir1[2], xrdir2[2]]; marker = :xcross)
        text!(xrdir1[1] - 0.06, xrdir1[2] - 0.06; text = "(", color = :gray70, fontsize = 20, rotation = -π / 4, font = :bold)
        text!(xrdir2[1] + 0.06, xrdir2[2] + 0.06; text = "(", color = :gray70, fontsize = 20, rotation = 3π / 4, font = :bold)

        xstar = res
        scatter!([xstar[1]], [xstar[2]]; marker = :circle, markersize = 15, strokewidth = 1, color = :black, strokecolor = :black)
        text!(xstar[1] + 0.04, xstar[2] + 0.04; text = L"x_{\star}", fontsize = 25, color = :black)
        ################################################
        return fig
    end
end

function plot_iterates_rayleigh(S::AbstractDFSolver, BP::AbstractBenchmarkProblem; invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(get_equality_manifold(BP), default_retraction_method(get_equality_manifold(BP))))
    palette = Makie.wong_colors()

    return with_theme(theme_latexfonts()) do
        res, ed, ih, oh, vs, ds = solve_problem(S, BP; invertibility_bound = invertibility_bound)
        fig = Figure(size = (500, 500))
        Axis(fig[1, 1], limits = ((0.1, 1.2), (0.1, 1.2)), bottomspinevisible = false, leftspinevisible = false, rightspinevisible = false, topspinevisible = false)
        hidedecorations!()

        # Contour lines
        f(x, y) = eval_obj(BP, [x, y])
        xs = LinRange(0.1, 1.2, 200)
        ys = LinRange(0.1, 1.2, 200)
        zs = [f(x, y) for x in xs, y in ys]
        contour!(xs, ys, zs; colormap = :thermometer, levels = -3.85:0.1:-3.25, labels = true)

        # Circle
        arc!(Point2f(0), 1, -π, π; color = :black)

        # Iterates
        data0 = ih[1]
        x0 = data0[1, :]
        scatter!([x0[1]], [x0[2]]; marker = :circle, color = palette[2], markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(x0[1] + 0.03, x0[2] + 0.03; text = L"x_{0}", fontsize = 25, color = palette[2])
        scatter!(data0[2:end, 1], data0[2:end, 2]; marker = :circle, markersize = 10, color = palette[2], strokewidth = 0)

        data1 = ih[2]
        x1 = data1[1, :]
        scatter!([x1[1]], [x1[2]]; marker = :circle, color = palette[3], markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(x1[1] + 0.01, x1[2] + 0.03; text = L"x_{1}", fontsize = 25, color = palette[3])
        scatter!(data1[2:end, 1], data1[2:end, 2]; marker = :circle, markersize = 10, color = palette[3], strokewidth = 0)

        scatter!([res[1]], [res[2]]; marker = :circle, markersize = 15, color = :black, strokecolor = :black, strokewidth = 1)
        text!(res[1] + 0.01, res[2] + 0.03; text = L"x_{\star}", fontsize = 25, color = :black)

        return fig
    end
end

function plot_iterates_lin_exp(S::AbstractDFSolver, BP::AbstractBenchmarkProblem; invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(get_equality_manifold(BP), default_retraction_method(get_equality_manifold(BP))))
    palette = Makie.wong_colors()

    return with_theme(theme_latexfonts()) do
        res, ed, ih, oh, vs, ds = solve_problem(S, BP; invertibility_bound = invertibility_bound)
        fig = Figure(size = (500, 1000))
        Axis(fig[1, 1], limits = ((-1.5, 1.5), (-4.5, 4.5)), bottomspinevisible = false, leftspinevisible = false, rightspinevisible = false, topspinevisible = false, aspect = DataAspect())
        hidedecorations!()

        # Contour lines
        f(x, y) = eval_obj(BP, [x, y])
        xs = LinRange(-1.5, 1.5, 200)
        ys = LinRange(-4.0, 4.0, 200)
        zs = [f(x, y) for x in xs, y in ys]
        contour!(xs, ys, zs; colormap = :thermometer, levels = -7:1:7, labels = true)

        # Circle
        arc!(Point2f(0), 1, -π, π; color = :black)

        it = 1
        it_data = ih[it]
        it_color = palette[mod(it, 7) + 1]
        xi = it_data[1, :]

        base_point = xi
        M = get_equality_manifold(BP)
        tdir1 = get_vector(M, base_point, [-4.2])
        tdir2 = get_vector(M, base_point, [4.2])
        xtdir1, xtdir2 = base_point .+ tdir1, base_point .+ tdir2
        lines!([xtdir1[1], xtdir2[1]], [xtdir1[2], xtdir2[2]]; color = :gray70)

        ##################### X0 ######################################
        # lines!([xtdir1[1], xtdir2[1]], [xtdir1[2], xtdir2[2]]; color = :gray70)
        scatter!([xi[1]], [xi[2]]; marker = :circle, color = it_color, markersize = 15, strokecolor = :black, strokewidth = 1)
        text!(xi[1] + 0.1, xi[2] + 0.03; text = L"x_{%$(it-1)}", fontsize = 30, color = it_color)
        text!(xtdir2[1] - 0.65, xtdir2[2] - 0.4; text = L"\mathrm{T}_{x_0}\mathbb{S}^1", fontsize = 30, color = :gray70)

        d = ds[it]
        bpd = base_point .+ transpose(d)
        radius = invertibility_radius(M, base_point, default_retraction_method(M), invertibility_bound)
        rdir1 = get_vector(M, base_point, [radius])
        rdir2 = get_vector(M, base_point, [-radius])
        xrdir1, xrdir2 = base_point .+ rdir1, base_point .+ rdir2
        text!(xrdir1[1] - 0.15, xrdir1[2] + 0.05; text = "(", color = :gray70, fontsize = 30, rotation = -π / 2, font = :bold)
        text!(xrdir2[1] + 0.15, xrdir2[2] - 0.05; text = "(", color = :gray70, fontsize = 30, rotation = π / 2, font = :bold)

        total_its = size(d)[1]
        ids_inside_radius = Int[]
        for i in 2:total_its
            di = d[i, :]
            if norm(di) < radius
                push!(ids_inside_radius, i)
                lines!([base_point[1] + di[1], it_data[i, 1]], [base_point[2] + di[2], it_data[i, 2]]; linestyle = :dot, color = it_color)
            end
        end
        ds_inside_radius = base_point .+ transpose(d[ids_inside_radius, :])
        Pds_inside_radius = it_data[ids_inside_radius, :]
        scatter!(Pds_inside_radius[:, 1], Pds_inside_radius[:, 2]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        scatter!(ds_inside_radius[1, :], ds_inside_radius[2, :]; marker = :circle, markersize = 10, color = it_color, strokewidth = 0)
        ################################################

        scatter!([res[1]], [res[2]]; marker = :circle, markersize = 15, strokewidth = 1, color = :black, strokecolor = :black)
        text!(res[1] - 0.25, res[2] - 0.25; text = L"x_{\star}", fontsize = 30, color = :black)
        return fig
    end
end

function plot_iterates_toy(S::AbstractDFSolver, BP::AbstractBenchmarkProblem; invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(get_equality_manifold(BP), default_retraction_method(get_equality_manifold(BP))))
    palette = Makie.wong_colors()

    return with_theme(theme_latexfonts()) do
        res, ed, ih, oh, vs, ds, xs = solve_problem(S, BP; invertibility_bound = invertibility_bound)
        fig = Figure(size = (500, 500))
        Axis(fig[1, 1], limits = ((-1.2, 1.2), (-1.2, 1.2)), bottomspinevisible = false, leftspinevisible = false, rightspinevisible = false, topspinevisible = false, aspect = DataAspect())
        hidedecorations!()

        # Contour lines
        f(x, y) = eval_obj(BP, [x, y])
        xs = LinRange(-1.2, 1.2, 200)
        ys = LinRange(-1.2, 1.2, 200)
        zs = [f(x, y) for x in xs, y in ys]
        contour!(xs, ys, zs; colormap = :thermometer, levels = -3:1:3, labels = true)

        # Circle
        arc!(Point2f(0), 1, -π, π; color = :black)
        text!(0.75, 0.75; text = L"\Omega_=", fontsize = 25, color = :black)

        n_outer_it = length(ih)

        for l in 1:n_outer_it
            it_data = ih[l]
            scatter!(it_data[:, 1], it_data[:, 2]; marker = :circle, markersize = 15, color = palette[1])
        end

        scatter!([res[1]], [res[2]]; marker = :circle, markersize = 20, color = palette[2])
        text!([res[1] - 0.13], [res[2] - 0.23]; text = L"x^*", fontsize = 25, color = palette[2])

        return fig
    end
end

function plot_iterates_toy_cobyla(BP::AbstractBenchmarkProblem)
    palette = Makie.wong_colors()

    return with_theme(theme_latexfonts()) do
        filename = "/home/sblelong/.julia/dev/ConstrainedDFO/tmp-cobyla.log"
        redirect_to_files(filename) do
            f(x) = eval_obj(BP, x)
            h(x) = eval_eqs(BP, x)
            x0 = get_x0(BP)
            result, solver_status = cobyla(f, x0; iprint = PRIMA.MSG_FEVL, xl = [-1.1, -1.1], xu = [1.1, 1.1], nonlinear_eq = h)
        end

        iterates = Vector{Float64}[]

        open(filename, "r") do logf
            for line in eachline(logf)
                if occursin("The corresponding X", line)
                    parts = split(line)
                    x = parse.(Float64, parts[(end - 1):end])
                    push!(iterates, x)
                end
            end
        end

        fig = Figure(size = (500, 500))
        Axis(fig[1, 1], limits = ((-1.2, 1.2), (-1.2, 1.2)), bottomspinevisible = false, leftspinevisible = false, rightspinevisible = false, topspinevisible = false, aspect = DataAspect())
        hidedecorations!()

        # Contour lines
        f(x, y) = eval_obj(BP, [x, y])
        xs = LinRange(-1.2, 1.2, 200)
        ys = LinRange(-1.2, 1.2, 200)
        zs = [f(x, y) for x in xs, y in ys]
        contour!(xs, ys, zs; colormap = :thermometer, levels = -3:1:3, labels = true)

        # Circle
        arc!(Point2f(0), 1, -π, π; color = :black)
        text!(0.75, 0.75; text = L"\Omega_=", fontsize = 25, color = :black)

        n_iterates = length(iterates)

        for k in 1:n_iterates
            xk = iterates[k]
            scatter!([xk[1]], [xk[2]]; marker = :circle, markersize = 15, color = palette[1])
        end

        res = iterates[end]
        scatter!([res[1]], [res[2]]; marker = :circle, markersize = 20, color = palette[2])
        text!([res[1] - 0.13], [res[2] - 0.23]; text = L"x^*", fontsize = 25, color = palette[2])

        return fig
    end
end

export plot_iterates, plot_iterates_rayleigh, plot_iterates_lin_exp, plot_iterates_toy, plot_iterates_toy_cobyla
