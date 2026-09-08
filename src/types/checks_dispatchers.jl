# These dispatchers are meant to patch the impossibility for `is_point` in `ManifoldsBase` to use a tolerance.

function is_point_dispatcher(M::AbstractManifold, p; tol_eqs::Float64 = 1.0e-8) end

is_point_dispatcher(M::Sphere, p; tol_eqs::Float64 = 1.0e-8) = is_point(M, p)
is_point_dispatcher(M::EqualityManifold, p; tol_eqs::Float64 = 1.0e-8) = is_point(M, p; tol_eqs = tol_eqs)
