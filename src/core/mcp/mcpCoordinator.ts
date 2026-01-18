import { SmartComposerSettings } from '../../settings/schema/setting.types'

import { McpManager } from './mcpManager'

type McpCoordinatorDeps = {
  getSettings: () => SmartComposerSettings
  registerSettingsListener: (
    listener: (settings: SmartComposerSettings) => void,
  ) => () => void
}

export class McpCoordinator {
  private static getMcpManagerCallCount = 0
  private readonly getSettings: () => SmartComposerSettings
  private readonly registerSettingsListener: (
    listener: (settings: SmartComposerSettings) => void,
  ) => () => void

  private mcpManager: McpManager | null = null
  private mcpManagerInitPromise: Promise<McpManager> | null = null
  private coordinatorId = Math.random().toString(36).substring(7)

  constructor(deps: McpCoordinatorDeps) {
    this.getSettings = deps.getSettings
    this.registerSettingsListener = deps.registerSettingsListener
  }

  async getMcpManager(): Promise<McpManager> {
    McpCoordinator.getMcpManagerCallCount++
    const callNum = McpCoordinator.getMcpManagerCallCount
    const timestamp = new Date().toISOString()
    const stack = new Error().stack

    console.debug(
      `[MCP-DEBUG] McpCoordinator.getMcpManager() CALLED #${callNum} at ${timestamp}`,
      {
        coordinatorId: this.coordinatorId,
        hasManager: !!this.mcpManager,
        hasInitPromise: !!this.mcpManagerInitPromise,
        callStack: stack,
      },
    )

    if (this.mcpManager) {
      console.debug(
        `[MCP-DEBUG] McpCoordinator.getMcpManager() #${callNum} Returning existing manager`,
      )
      return this.mcpManager
    }

    if (!this.mcpManagerInitPromise) {
      console.debug(
        `[MCP-DEBUG] McpCoordinator.getMcpManager() #${callNum} Creating NEW McpManager`,
      )
      this.mcpManagerInitPromise = (async () => {
        try {
          console.debug(
            `[MCP-DEBUG] McpCoordinator.getMcpManager() #${callNum} Constructing McpManager`,
          )
          this.mcpManager = new McpManager({
            settings: this.getSettings(),
            registerSettingsListener: this.registerSettingsListener,
          })
          console.debug(
            `[MCP-DEBUG] McpCoordinator.getMcpManager() #${callNum} Calling initialize()`,
          )
          await this.mcpManager.initialize()
          console.debug(
            `[MCP-DEBUG] McpCoordinator.getMcpManager() #${callNum} Initialize COMPLETED`,
          )
          return this.mcpManager
        } catch (error) {
          console.error(
            `[MCP-DEBUG] McpCoordinator.getMcpManager() #${callNum} Initialize FAILED`,
            error,
          )
          this.mcpManager = null
          this.mcpManagerInitPromise = null
          throw error
        }
      })()
    } else {
      console.debug(
        `[MCP-DEBUG] McpCoordinator.getMcpManager() #${callNum} Waiting for existing initialization promise`,
      )
    }

    return this.mcpManagerInitPromise
  }

  cleanup() {
    const timestamp = new Date().toISOString()
    const stack = new Error().stack

    console.debug(
      `[MCP-DEBUG] McpCoordinator.cleanup() CALLED at ${timestamp}`,
      {
        coordinatorId: this.coordinatorId,
        hasManager: !!this.mcpManager,
        hasInitPromise: !!this.mcpManagerInitPromise,
        callStack: stack,
      },
    )

    if (this.mcpManager) {
      console.debug(
        `[MCP-DEBUG] McpCoordinator.cleanup() Calling McpManager.cleanup()`,
      )
      this.mcpManager.cleanup()
    }
    this.mcpManager = null
    this.mcpManagerInitPromise = null

    console.debug(
      `[MCP-DEBUG] McpCoordinator.cleanup() COMPLETED at ${new Date().toISOString()}`,
    )
  }
}
