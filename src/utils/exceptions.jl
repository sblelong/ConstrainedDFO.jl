struct NumericalError <: Exception
    msg::String
end

showerror(io::IO, e::NumericalError) = print(e.msg)
