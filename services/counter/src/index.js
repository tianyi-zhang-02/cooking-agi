/* Cloudflare entry: a stateless Worker in front of one Durable Object that
   holds every counter (SQLite storage, available on the Workers Free plan).
   All logic lives in core.js; this file only wires it to the platform. */
import {DurableObject} from 'cloudflare:workers';
import {CounterCore, Catalog, handle} from './core.js';

export class Counters extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.core = new CounterCore(ctx.storage.sql);
    ctx.blockConcurrencyWhile(() => this.core.init());
  }
  presence(beat, client) { return this.core.presence(beat, client); }
  snapshot(note, client) { return this.core.snapshot(note, client); }
  record(note, kind, token, saved, client) { return this.core.record(note, kind, token, saved, client); }
}

let catalog = null;
export default {
  fetch(request, env) {
    catalog ??= new Catalog(env.CATALOG_URL);
    const counters = env.COUNTERS.get(env.COUNTERS.idFromName('global'));
    return handle(request, env, counters, catalog);
  }
};
