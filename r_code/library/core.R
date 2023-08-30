#' pkg_install_load
#'
#' Load a list of packages. Packages are installed if required.
#'
#' @param packages  a comma separated list of packages
#'
#' @export
#'
#' @examples
#' # load single package
#' pkg_install_load("e1071")
#' # load multiple packages
#' pkg_install_load("e1071", "caret", "rminer")
pkg_install_load <- function(package1, ...) {
   packages <- c(package1, ...)
   for (package in packages) {
      if (package %in% rownames(installed.packages())) {
         suppressWarnings(suppressMessages(do.call("library", list(package))))
      } else {
         install.packages(package)
         suppressWarnings(suppressMessages(do.call("library", list(package))))
      }
   }
}

#' call_without_warnings
#'
#' Call a function without displaying the warning message
#'
#' @param fn  the function to be called
#' @param fn_args  a comma separated list of arguments to be passed
#'                 to the function
#'
#' @return the result of the function
#' @export
#'
#' @examples
#' # call single function
#' my_fn <- function() {
#'    print("hello")
#' }
#' call_without_warnings(my_fn)
#' # call function with arguments
#' my_fn <- function(param) {
#'    print(param)
#' }
#' call_without_warnings(my_fn, param = "hello")
#' # get value returned by function
#' my_fn <- function(x) {
#'    return(x^2)
#' }
#' result <- call_without_warnings(my_fn, x = 5)
call_without_warnings <- function(fn, ...) {
   warns <- getOption("warn")
   options(warn = -1)
   result <- fn(...)
   options(warn = warns)
   return(result)
}
