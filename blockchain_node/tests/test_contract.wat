(module
  (memory 1)
  (export "memory" (memory 0))

  (func $test_function (param i32) (result i32)
    local.get 0
    i32.const 1
    i32.add
  )

  (func $recursive_function (param i32) (result i32)
    local.get 0
    i32.const 0
    i32.eq
    if (result i32)
      i32.const 1
    else
      local.get 0
      i32.const 1
      i32.sub
      call $recursive_function
      local.get 0
      i32.mul
    end
  )

  (func $complex_function (param i32 i32) (result i32)
    (local i32 i32)
    local.get 0
    local.set 2
    local.get 1
    local.set 3
    block
      loop
        local.get 2
        i32.const 0
        i32.eq
        br_if 1
        local.get 2
        i32.const 1
        i32.sub
        local.set 2
        local.get 3
        i32.const 1
        i32.add
        local.set 3
        br 0
      end
    end
    local.get 3
  )

  (export "test_function" (func $test_function))
  (export "recursive_function" (func $recursive_function))
  (export "complex_function" (func $complex_function))
) 