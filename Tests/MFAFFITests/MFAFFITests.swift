import XCTest
@testable import MFAFFI

final class MFAFFITests: XCTestCase {
    func testContextCreation() throws {
        var context: UnsafeMutableRawPointer?
        let result = mfa_create_context(&context)

        XCTAssertEqual(result, MFA_SUCCESS, "Context creation should succeed")
        XCTAssertNotNil(context, "Context should not be nil")

        if let context = context {
            mfa_destroy_context(context)
        }
    }

    func testDeviceSupport() throws {
        let isSupported = mfa_is_device_supported()
        XCTAssertTrue(isSupported, "Metal device should be supported on Apple platforms")
    }

    func testVersion() throws {
        var major: Int32 = 0
        var minor: Int32 = 0
        var patch: Int32 = 0

        mfa_get_version(&major, &minor, &patch)

        XCTAssertEqual(major, 1)
        XCTAssertEqual(minor, 0)
        XCTAssertEqual(patch, 0)
    }

    func testErrorStrings() throws {
        let successStr = mfa_error_string(MFA_SUCCESS)
        let invalidArgsStr = mfa_error_string(MFA_ERROR_INVALID_ARGS)

        XCTAssertNotNil(successStr)
        XCTAssertNotNil(invalidArgsStr)

        if let successStr = successStr {
            let swiftStr = String(cString: successStr)
            XCTAssertEqual(swiftStr, "Success")
            free(UnsafeMutableRawPointer(mutating: successStr))
        }

        if let invalidArgsStr = invalidArgsStr {
            let swiftStr = String(cString: invalidArgsStr)
            XCTAssertEqual(swiftStr, "Invalid arguments")
            free(UnsafeMutableRawPointer(mutating: invalidArgsStr))
        }
    }

    func testBufferManagement() throws {
        var context: UnsafeMutableRawPointer?
        let contextResult = mfa_create_context(&context)
        XCTAssertEqual(contextResult, MFA_SUCCESS)

        defer {
            if let context = context {
                mfa_destroy_context(context)
            }
        }

        guard let context = context else {
            XCTFail("Context creation failed")
            return
        }

        // Test buffer creation
        var buffer: UnsafeMutableRawPointer?
        let bufferResult = mfa_create_buffer(context, 1024, &buffer)
        XCTAssertEqual(bufferResult, MFA_SUCCESS, "Buffer creation should succeed")
        XCTAssertNotNil(buffer, "Buffer should not be nil")

        // Test buffer contents access
        if let buffer = buffer {
            let contents = mfa_buffer_contents(buffer)
            XCTAssertNotNil(contents, "Buffer contents should be accessible")
            mfa_destroy_buffer(buffer)
        }
    }
}