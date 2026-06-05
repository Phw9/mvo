#include "mvo/app.h"

int main(int argc, char** argv) {
    const mvo::AppConfig config = mvo::parse_args(argc, argv);
    const int32_t exit_code = mvo::run_app(config);
    return exit_code;
}
