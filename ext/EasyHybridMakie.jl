module EasyHybridMakie

using EasyHybrid
using Makie
import Makie
@info "Extension loaded!"

Makie.convert_single_argument(v::WrappedTuples) = Matrix(v)

end